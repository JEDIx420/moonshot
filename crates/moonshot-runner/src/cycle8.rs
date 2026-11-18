use crate::cycle7::{hash, jsonl, redact, text_features, write_json, Ranker, WIDTH};
use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    tensor::{Bool, Device, Int, Tensor, TensorData},
};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SLOTS: usize = 16;
const NONE: usize = SLOTS * 3;
const BATCH: usize = 32;
const STEPS: usize = 6000;
const LR: f64 = 0.001;
const RELATIONS: [&str; 3] = ["supports", "contradicts", "insufficient"];
const CONDITIONS: [&str; 3] = ["tiny_fresh", "query_blind", "shuffled_labels"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case_id: String,
    query: Query,
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    text: String,
    claim_call_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    id: String,
    tool_name: String,
    call_id: Option<String>,
    status: String,
    text: String,
}

#[derive(Deserialize)]
struct Answer {
    case_id: String,
    candidate_id: String,
    relation: String,
    reviewed_by: String,
}

#[derive(Debug, Serialize)]
struct Prediction<'a> {
    case_id: &'a str,
    ranked_candidate_ids: Vec<&'a str>,
    predicted_relation: &'static str,
    confidence: f64,
    abstain: bool,
}

fn read_cases(path: &Path) -> Result<Vec<Case>> {
    let cases: Vec<Case> = jsonl(path)?;
    let mut ids = HashSet::new();
    for case in &cases {
        let mut candidates = HashSet::new();
        if case.case_id.trim().is_empty()
            || !ids.insert(&case.case_id)
            || case.query.text.trim().is_empty()
            || !(2..=SLOTS).contains(&case.candidates.len())
        {
            return Err("invalid case ID, claim, or candidate count".into());
        }
        for item in &case.candidates {
            if item.id == "NONE"
                || item.id.trim().is_empty()
                || !candidates.insert(&item.id)
                || item.tool_name.trim().is_empty()
                || item.text.trim().is_empty()
                || !matches!(item.status.as_str(), "ok" | "error" | "pending" | "unknown")
            {
                return Err("invalid candidate record".into());
            }
        }
    }
    if cases.is_empty() {
        return Err("empty input partition".into());
    }
    Ok(cases)
}

fn labels(cases: &[Case], path: &Path) -> Result<Vec<i64>> {
    let answers: Vec<Answer> = jsonl(path)?;
    if answers.len() != cases.len() {
        return Err("training/validation answers are incomplete".into());
    }
    let mut by_id: HashMap<_, _> = answers
        .into_iter()
        .map(|answer| (answer.case_id.clone(), answer))
        .collect();
    cases
        .iter()
        .map(|case| {
            let gold = by_id.remove(&case.case_id).ok_or("missing answer ID")?;
            if gold.reviewed_by.trim().is_empty() {
                return Err("missing recorded reviewer".into());
            }
            if gold.candidate_id == "NONE" && gold.relation == "insufficient" {
                return Ok(NONE as i64);
            }
            let slot = case
                .candidates
                .iter()
                .position(|c| c.id == gold.candidate_id)
                .ok_or("gold candidate is absent")?;
            let class = RELATIONS
                .iter()
                .position(|&r| r == gold.relation)
                .ok_or("unknown gold relation")?;
            Ok((slot * 3 + class) as i64)
        })
        .collect()
}

fn features(case: &Case, blind: bool) -> Vec<f32> {
    let mut values = vec![0.0; SLOTS * WIDTH];
    for (event, row) in case.candidates.iter().zip(values.chunks_exact_mut(WIDTH)) {
        if !blind {
            row[0] = f32::from(
                case.query
                    .claim_call_id
                    .as_ref()
                    .zip(event.call_id.as_ref())
                    .is_some_and(|(a, b)| !a.is_empty() && a == b),
            );
            text_features(&redact(&case.query.text), &mut row[16..256]);
        }
        for (index, status) in ["ok", "error", "pending", "unknown"].iter().enumerate() {
            row[index + 1] = f32::from(event.status == *status);
        }
        text_features(&redact(&event.tool_name), &mut row[256..304]);
        text_features(&redact(&event.text), &mut row[304..]);
    }
    values
}

fn forward(model: &Ranker, rows: &[(&Case, &Vec<f32>)], device: &Device) -> Tensor<2> {
    let batch = rows.len();
    let input = Tensor::from_data(
        TensorData::new(
            rows.iter()
                .flat_map(|(_, values)| values.iter().copied())
                .collect::<Vec<_>>(),
            [batch, SLOTS, WIDTH],
        ),
        device,
    );
    let valid: Vec<_> = rows
        .iter()
        .flat_map(|(case, _)| (0..SLOTS).map(|slot| slot < case.candidates.len()))
        .collect();
    let valid = Tensor::<2, Bool>::from_data(TensorData::new(valid, [batch, SLOTS]), device);
    let scores = model.pair_scores(input);
    let invalid = valid.clone().bool_not().reshape([batch, SLOTS, 1]);
    let pair_logits = scores
        .clone()
        .slice([0..batch, 0..SLOTS, 0..3])
        .mask_fill(invalid.clone(), -1e9)
        .reshape([batch, NONE]);
    let none = scores
        .slice([0..batch, 0..SLOTS, 5..6])
        .mask_fill(invalid, 0.0)
        .sum_dim(1)
        .reshape([batch, 1])
        / valid.float().sum_dim(1);
    Tensor::cat(vec![pair_logits, none], 1)
}

fn decode<'a>(case: &'a Case, logits: &[f32]) -> (usize, Prediction<'a>) {
    let len = case.candidates.len();
    assert!(logits.iter().all(|v| v.is_finite()));
    let mut order: Vec<_> = (0..len)
        .map(|slot| {
            (slot * 3..slot * 3 + 3)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap()
        })
        .chain([NONE])
        .collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let best = order[0];
    let confidence = 1.0
        / (0..len * 3)
            .chain([NONE])
            .map(|index| f64::from(logits[index] - logits[best]).exp())
            .sum::<f64>();
    (
        best,
        Prediction {
            case_id: &case.case_id,
            ranked_candidate_ids: order
                .into_iter()
                .filter(|&index| index != NONE || best == NONE)
                .map(|index| {
                    if index == NONE {
                        "NONE"
                    } else {
                        &case.candidates[index / 3].id
                    }
                })
                .collect(),
            predicted_relation: if best == NONE {
                "insufficient"
            } else {
                RELATIONS[best % 3]
            },
            confidence,
            abstain: false,
        },
    )
}

fn predict<'a>(
    model: &Ranker,
    cases: &'a [Case],
    data: &[Vec<f32>],
    device: &Device,
) -> Result<Vec<(usize, Prediction<'a>)>> {
    let mut result = Vec::with_capacity(cases.len());
    for (cases, data) in cases.chunks(BATCH).zip(data.chunks(BATCH)) {
        let rows: Vec<_> = cases.iter().zip(data).collect();
        let logits = forward(model, &rows, device)
            .to_data_as::<f32>()
            .try_into_vec::<f32>()?;
        result.extend(
            cases
                .iter()
                .zip(logits.chunks_exact(NONE + 1))
                .map(|(case, logits)| decode(case, logits)),
        );
    }
    Ok(result)
}

fn choose_cutoff(predictions: &[(usize, Prediction<'_>)], targets: &[i64]) -> Option<f64> {
    let mut ranked: Vec<_> = predictions
        .iter()
        .zip(targets)
        .filter(|((_, pred), _)| pred.predicted_relation != "insufficient")
        .map(|((class, pred), &label)| (pred.confidence, *class as i64 == label))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut best = None;
    let mut correct = 0;
    for (i, &(confidence, hit)) in ranked.iter().enumerate() {
        correct += usize::from(hit);
        let count = i + 1;
        if ranked.get(count).is_none_or(|next| next.0 != confidence) && correct * 100 >= count * 95
        {
            best = Some(confidence);
        }
    }
    best
}

fn save_predictions(
    path: &Path,
    mut predictions: Vec<(usize, Prediction<'_>)>,
    cutoff: Option<f64>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for (_, prediction) in &mut predictions {
        prediction.abstain = cutoff.is_none_or(|c| prediction.confidence < c);
        serde_json::to_writer(&mut writer, prediction)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn run(
    seed: u64,
    dataset: &Path,
    commit: &str,
    output: &Path,
    fresh_only: bool,
) -> Result<()> {
    if !fresh_only {
        return Err("compact_pretrained is unavailable; this build only supports an explicitly scoped --fresh-only run".into());
    }
    if ![8001, 8109, 8207, 8303].contains(&seed)
        || commit.len() != 40
        || !commit.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid preregistered seed or source SHA".into());
    }
    if dataset.join("answers/test.jsonl").exists() {
        return Err("sealed test answers are present; refusing initialization".into());
    }
    let train = read_cases(&dataset.join("inputs/train.jsonl"))?;
    let validation = read_cases(&dataset.join("inputs/validation.jsonl"))?;
    let original = labels(&train, &dataset.join("answers/train.jsonl"))?;
    let gold = labels(&validation, &dataset.join("answers/validation.jsonl"))?;
    let test_path = dataset.join("inputs/test.jsonl");
    let mut hashes = serde_json::Map::new();
    for filename in [
        "inputs/train.jsonl",
        "inputs/validation.jsonl",
        "inputs/test.jsonl",
        "answers/train.jsonl",
        "answers/validation.jsonl",
    ] {
        hashes.insert(
            filename.into(),
            hash(&fs::read(dataset.join(filename))?).into(),
        );
    }
    fs::create_dir_all(output)?;
    let gpu = Device::cuda(0);
    let ad = gpu.clone().autodiff();
    ad.seed(seed);
    let initial_model = Ranker::new(&ad);
    let init = initial_model.checkpoint()?;
    let parameters = initial_model.num_params();
    let mut updates = serde_json::Map::new();
    let mut cutoffs = HashMap::new();
    for condition in CONDITIONS {
        let dir = output.join(condition);
        fs::create_dir(&dir)?;
        let blind = condition == "query_blind";
        let train_data: Vec<_> = train.iter().map(|case| features(case, blind)).collect();
        let val_data: Vec<_> = validation
            .iter()
            .map(|case| features(case, blind))
            .collect();
        let mut target = original.clone();
        if condition == "shuffled_labels" {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 808808);
            for count in 2..=SLOTS {
                let indices: Vec<_> = train
                    .iter()
                    .enumerate()
                    .filter(|(_, case)| case.candidates.len() == count)
                    .map(|(i, _)| i)
                    .collect();
                let mut shuffled: Vec<_> = indices.iter().map(|&i| target[i]).collect();
                shuffled.shuffle(&mut rng);
                for (i, label) in indices.into_iter().zip(shuffled) {
                    target[i] = label;
                }
            }
        }
        let mut model = Ranker::restore(&init, &ad)?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 800800);
        let mut optimizer = AdamConfig::new().init();
        let loss_fn = CrossEntropyLossConfig::new().init(&ad);
        let mut curve = BufWriter::new(File::create(dir.join("training_curve.csv"))?);
        writeln!(curve, "update,examples,loss,elapsed_seconds")?;
        let mut best = None;
        let mut candidates = Vec::new();
        let start = Instant::now();
        for step in 1..=STEPS {
            let idx: Vec<_> = (0..BATCH).map(|_| rng.gen_range(0..train.len())).collect();
            let rows: Vec<_> = idx.iter().map(|&i| (&train[i], &train_data[i])).collect();
            let y = Tensor::<1, Int>::from_data(
                TensorData::new(idx.iter().map(|&i| target[i]).collect::<Vec<_>>(), [BATCH]),
                &ad,
            );
            let loss = loss_fn.forward(forward(&model, &rows, &ad), y);
            let observed = loss.clone().into_scalar::<f32>();
            assert!(observed.is_finite(), "nonfinite loss");
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            assert_eq!(gradients.len(), 4, "missing gradient tensors");
            model = optimizer.step(LR, model, gradients);
            writeln!(
                curve,
                "{step},{},{observed},{}",
                step * BATCH,
                start.elapsed().as_secs_f64()
            )?;
            if step % 500 == 0 {
                curve.flush()?;
                println!("cycle8 {seed} {condition} update {step}/{STEPS} loss {observed}");
            }
            if [1500, 3000, 6000].contains(&step) {
                let bytes = model.checkpoint()?;
                assert_ne!(hash(&bytes), hash(&init));
                let restored = Ranker::restore(&bytes, &gpu)?;
                let predictions = predict(&restored, &validation, &val_data, &gpu)?;
                let correct = predictions
                    .iter()
                    .zip(&gold)
                    .filter(|((class, _), label)| *class as i64 == **label)
                    .count();
                candidates.push(json!({"update":step,"validation_exact_count":correct}));
                if best.is_none_or(|previous| correct > previous) {
                    best = Some(correct);
                    let n = BATCH.min(validation.len());
                    let probe = predict(&model, &validation[..n], &val_data[..n], &ad)?;
                    assert!(
                        probe.iter().zip(&predictions).all(|(a, b)| a.0 == b.0
                            && a.1.ranked_candidate_ids == b.1.ranked_candidate_ids
                            && (a.1.confidence - b.1.confidence).abs() < 1e-5),
                        "cold reload parity failed"
                    );
                    fs::write(dir.join("model.safetensors"), &bytes)?;
                    let cutoff = choose_cutoff(&predictions, &gold);
                    cutoffs.insert(condition, cutoff);
                    save_predictions(
                        &dir.join("validation_predictions.jsonl"),
                        predictions,
                        cutoff,
                    )?;
                    write_json(
                        &dir.join("selection.json"),
                        &json!({
                            "update":step,"validation_exact_count":correct,"validation_confidence_threshold":cutoff
                        }),
                    )?;
                    write_json(
                        &dir.join("checkpoint.json"),
                        &json!({
                            "sha256":hash(&bytes),"bytes":bytes.len(),"cold_reload_parity":true,
                            "probe_cases":n,"model_parameters":parameters
                        }),
                    )?;
                }
            }
        }
        curve.flush()?;
        write_json(&dir.join("validation_candidates.json"), &json!(candidates))?;
        updates.insert(condition.into(), STEPS.into());
    }
    let test = read_cases(&test_path)?;
    for condition in CONDITIONS {
        let dir = output.join(condition);
        let model = Ranker::restore(&fs::read(dir.join("model.safetensors"))?, &gpu)?;
        let data: Vec<_> = test
            .iter()
            .map(|case| features(case, condition == "query_blind"))
            .collect();
        let predictions = predict(&model, &test, &data, &gpu)?;
        save_predictions(
            &dir.join("test_predictions.jsonl"),
            predictions,
            cutoffs[condition],
        )?;
    }
    write_json(
        &output.join("run_manifest.json"),
        &json!({
            "seed":seed,"git_commit":commit,"device":"burn-cuda","status":"fresh_only_completed",
            "conditions":CONDITIONS,"missing_condition":"compact_pretrained","model_parameters":parameters,
            "updates_by_condition":updates,"dataset_hashes":hashes,"gradient_tensors_per_update":4,
            "batch_size":BATCH,"optimizer":"Adam","learning_rate":LR,
            "feature_schema":"redacted FNV-1a text buckets, recorded call equality and status, no raw IDs or candidate positions",
            "threshold_policy":"maximum validation coverage at >=95% exact assertive precision; null means abstain"
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle8_masks_padding_and_preserves_candidate_relation() {
        let case: Case = serde_json::from_value(json!({
            "case_id":"test", "query":{"text":"report queued","claim_call_id":"x"},
            "candidates":[
                {"id":"e1","tool_name":"lookup","call_id":null,"status":"unknown","text":"found"},
                {"id":"e0","tool_name":"queue","call_id":"x","status":"pending","text":"queued"}
            ]
        }))
        .unwrap();
        let data = features(&case, false);
        assert_eq!((data[0], data[WIDTH]), (0.0, 1.0));
        assert_eq!(features(&case, true)[WIDTH], 0.0);
        let cpu = Device::flex();
        let model = Ranker::new(&cpu);
        let logits = forward(&model, &[(&case, &data)], &cpu)
            .to_data_as::<f32>()
            .try_into_vec::<f32>()
            .unwrap();
        assert_eq!(logits.len(), NONE + 1);
        assert!(logits[6..NONE].iter().all(|&value| value == -1e9));
        let mut logits = vec![0.0; NONE + 1];
        logits[7] = 20.0;
        logits[4] = 9.0;
        let (class, pred) = decode(&case, &logits);
        assert_eq!(class, 4);
        assert_eq!(pred.ranked_candidate_ids, ["e0", "e1"]);
        assert_eq!(pred.predicted_relation, "contradicts");
    }
}
