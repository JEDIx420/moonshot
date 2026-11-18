use burn::module::{Module, Param};
use burn::nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Int, Tensor, TensorData};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const WIDTH: usize = 512;
const HIDDEN: usize = 1920;
const BATCH: usize = 64;
const STEPS: usize = 12000;
const LR: f64 = 0.001;
const REL: [&str; 5] = [
    "supports",
    "contradicts",
    "identity",
    "divide_100",
    "multiply_100",
];
const NONE: usize = 20;
const CONDITIONS: [&str; 5] = [
    "text_only",
    "structure_only",
    "hybrid",
    "query_blind",
    "shuffled_labels",
];

#[derive(Deserialize)]
struct Case {
    case_id: String,
    track: String,
    query: Value,
    candidates: [Candidate; 4],
}

#[derive(Deserialize)]
struct Candidate {
    id: String,
    text: String,
    data: Value,
}

#[derive(Deserialize)]
struct Answer {
    case_id: String,
    candidate_id: String,
    relation: String,
}

#[derive(Serialize, PartialEq, Debug)]
struct Prediction<'a> {
    case_id: &'a str,
    ranked_candidate_ids: Vec<&'a str>,
    predicted_relation: &'static str,
    confidence: f64,
    abstain: bool,
}

#[derive(Module, Debug)]
struct Ranker {
    hidden: Linear,
    output: Linear,
}

impl Ranker {
    fn new(device: &Device) -> Self {
        Self {
            hidden: LinearConfig::new(WIDTH, HIDDEN).init(device),
            output: LinearConfig::new(HIDDEN, 6).init(device),
        }
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch, _, _] = input.dims();
        let scores = self
            .output
            .forward(activation::relu(self.hidden.forward(input)));
        let pairs = scores
            .clone()
            .slice([0..batch, 0..4, 0..5])
            .reshape([batch, 20]);
        let none = scores
            .slice([0..batch, 0..4, 5..6])
            .mean_dim(1)
            .reshape([batch, 1]);
        Tensor::cat(vec![pairs, none], 1)
    }

    fn checkpoint(&self) -> Result<Vec<u8>> {
        let data = [
            (
                "hidden.weight",
                vec![WIDTH, HIDDEN],
                self.hidden.weight.val().to_data_as::<f32>(),
            ),
            (
                "hidden.bias",
                vec![HIDDEN],
                self.hidden.bias.as_ref().unwrap().val().to_data_as::<f32>(),
            ),
            (
                "output.weight",
                vec![HIDDEN, 6],
                self.output.weight.val().to_data_as::<f32>(),
            ),
            (
                "output.bias",
                vec![6],
                self.output.bias.as_ref().unwrap().val().to_data_as::<f32>(),
            ),
        ];
        let tensors: Vec<_> = data
            .into_iter()
            .map(|(name, shape, data)| {
                let bytes: Vec<_> = data
                    .try_into_vec::<f32>()?
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect();
                if !bytes
                    .chunks_exact(4)
                    .all(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()).is_finite())
                {
                    return Err("nonfinite checkpoint weight".into());
                }
                Ok((name, shape, bytes))
            })
            .collect::<Result<_>>()?;
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                TensorView::new(Dtype::F32, shape.clone(), bytes).map(|view| (*name, view))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(serialize(views, &None)?)
    }

    fn restore(bytes: &[u8], device: &Device) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)?;
        let load = |name, shape: &[usize]| -> Result<Vec<f32>> {
            let tensor = tensors.tensor(name)?;
            if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
                return Err(format!("wrong checkpoint shape or dtype: {name}").into());
            }
            Ok(tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        };
        let layer = |weight, bias, inputs, outputs| -> Result<Linear> {
            let shape = [inputs, outputs];
            Ok(Linear {
                weight: Param::from_data(TensorData::new(load(weight, &shape)?, shape), device),
                bias: Some(Param::from_data(
                    TensorData::new(load(bias, &[outputs])?, [outputs]),
                    device,
                )),
            })
        };
        Ok(Self {
            hidden: layer("hidden.weight", "hidden.bias", WIDTH, HIDDEN)?,
            output: layer("output.weight", "output.bias", HIDDEN, 6)?,
        })
    }
}

fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn jsonl<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn text_features(text: &str, bins: &mut [f32]) {
    for token in text
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|token| !token.is_empty())
    {
        let hash = token.bytes().fold(2166136261u32, |state, byte| {
            (state ^ byte.to_ascii_lowercase() as u32).wrapping_mul(16777619)
        });
        let bin = hash as usize % bins.len();
        bins[bin] = 1.0;
    }
}

fn redact(text: &str) -> String {
    // Remove entire opaque ID bindings and time values before token hashing.
    text.split_whitespace()
        .filter(|word| !word.bytes().any(|b| b.is_ascii_digit()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn features(case: &Case, condition: &str) -> Vec<f32> {
    let mut values = vec![0.0; WIDTH * 4];
    for (candidate, row) in case.candidates.iter().zip(values.chunks_exact_mut(WIDTH)) {
        let is_agent = case.track == "agent_evidence";
        row[0] = f32::from(is_agent);
        if condition != "structure_only" {
            if condition != "query_blind" {
                text_features(
                    &redact(case.query["text"].as_str().unwrap()),
                    &mut row[32..272],
                );
            }
            text_features(&redact(&candidate.text), &mut row[272..]);
        }
        if matches!(condition, "hybrid" | "structure_only" | "shuffled_labels") {
            let data = &candidate.data;
            let query = &case.query;
            if is_agent {
                row[1] = f32::from(query["trace_id"] == data["trace_id"]);
                row[2] = f32::from(query["resource_id"] == data["resource_id"]);
                row[3] = f32::from(data["status"] == "ok");
                let age = query["timestamp_ms"].as_i64().unwrap()
                    - data["timestamp_ms"].as_i64().unwrap();
                row[4] = age.clamp(-1000, 1000) as f32 / 1000.0;
            } else {
                let source = &query["data"];
                row[5] = f32::from(source["scope"] == data["scope"]);
                row[6] = f32::from(source["dtype"] == data["dtype"]);
                row[7] = f32::from(source["unit"] == data["unit"]);
                for (i, unit) in ["native", "minor", "major"].iter().enumerate() {
                    row[8 + i] = f32::from(source["unit"] == *unit);
                    row[11 + i] = f32::from(data["unit"] == *unit);
                }
            }
        }
    }
    values
}

fn targets(cases: &[Case], answers: Vec<Answer>) -> Result<Vec<i64>> {
    let answer_count = answers.len();
    let mut by_id: std::collections::HashMap<_, _> = answers
        .into_iter()
        .map(|answer| (answer.case_id.clone(), answer))
        .collect();
    if cases.len() != answer_count || by_id.len() != answer_count {
        return Err("wrong answer count or duplicate ID".into());
    }
    cases
        .iter()
        .map(|case| {
            let answer = by_id.remove(&case.case_id).ok_or("missing answer ID")?;
            if answer.candidate_id == "NONE" && answer.relation == "none" {
                return Ok(NONE as i64);
            }
            let slot = case
                .candidates
                .iter()
                .position(|item| item.id == answer.candidate_id)
                .ok_or("unknown answer candidate")?;
            let rel = REL
                .iter()
                .position(|rel| *rel == answer.relation)
                .ok_or("unknown relation")?;
            Ok((slot * 5 + rel) as i64)
        })
        .collect()
}

fn tensor(data: Vec<f32>, len: usize, device: &Device) -> Tensor<3> {
    Tensor::from_data(TensorData::new(data, [len, 4, WIDTH]), device)
}

fn decode<'a>(case: &'a Case, logits: &[f32]) -> (usize, Prediction<'a>) {
    assert!(logits.iter().all(|value| value.is_finite()));
    let valid = if case.track == "agent_evidence" {
        0..2
    } else {
        2..5
    };
    let mut ranks: Vec<_> = (0..4)
        .map(|slot| {
            valid
                .clone()
                .map(|rel| slot * 5 + rel)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap()
        })
        .chain([NONE])
        .collect();
    ranks.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let best = ranks[0];
    let denom = (0..4)
        .flat_map(|slot| valid.clone().map(move |rel| slot * 5 + rel))
        .chain([NONE])
        .map(|i| f64::from(logits[i] - logits[best]).exp())
        .sum::<f64>();
    let ranked_candidate_ids = ranks
        .into_iter()
        .filter(|&i| i != NONE || best == NONE)
        .map(|i| {
            if i == NONE {
                "NONE"
            } else {
                &case.candidates[i / 5].id
            }
        })
        .collect();
    (
        best,
        Prediction {
            case_id: &case.case_id,
            ranked_candidate_ids,
            predicted_relation: if best == NONE { "none" } else { REL[best % 5] },
            confidence: 1.0 / denom,
            abstain: false,
        },
    )
}

fn predict<'a>(
    model: &Ranker,
    cases: &'a [Case],
    data: &[Vec<f32>],
    device: &Device,
) -> Vec<(usize, Prediction<'a>)> {
    cases
        .chunks(BATCH)
        .zip(data.chunks(BATCH))
        .flat_map(|(cases, batch)| {
            let logits = model
                .forward(tensor(
                    batch.iter().flatten().copied().collect(),
                    batch.len(),
                    device,
                ))
                .to_data_as::<f32>()
                .try_into_vec::<f32>()
                .unwrap();
            cases
                .iter()
                .zip(logits.chunks_exact(21))
                .map(|(case, logits)| decode(case, logits))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn predict_file(path: &Path, predictions: &[(usize, Prediction<'_>)]) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for (_, prediction) in predictions {
        serde_json::to_writer(&mut file, prediction)?;
        writeln!(file)?;
    }
    file.flush()?;
    Ok(())
}

fn threshold(
    cases: &[Case],
    predictions: &[(usize, Prediction<'_>)],
    targets: &[i64],
) -> (f64, bool) {
    let mut results: Vec<_> = cases
        .iter()
        .zip(predictions.iter().zip(targets))
        .filter(|(case, _)| case.track == "agent_evidence")
        .map(|(_, ((class, pred), label))| (pred.confidence, *class as i64 == *label))
        .collect();
    results.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut hits = 0;
    let mut best = (1.0, false);
    for (i, &(confidence, correct)) in results.iter().enumerate() {
        hits += usize::from(correct);
        let count = i + 1;
        let end_of_tie = results.get(count).is_none_or(|next| next.0 != confidence);
        if end_of_tie && 100 * hits >= 95 * count {
            best = (confidence, true);
        }
    }
    best
}

fn cpu_profile(checkpoint: &[u8], cases: &[Case], condition: &str) -> Result<Value> {
    let device = Device::flex();
    let model = Ranker::restore(checkpoint, &device)?;
    let mut latency = Vec::new();
    let mut preprocess = Vec::new();
    for (i, case) in cases.iter().step_by(12).take(256).enumerate() {
        let start = Instant::now();
        let data = features(case, condition);
        let prep_ms = start.elapsed().as_secs_f64() * 1000.0;
        let logits = model
            .forward(tensor(data, 1, &device))
            .to_data_as::<f32>()
            .try_into_vec::<f32>()?;
        std::hint::black_box(decode(case, &logits));
        if i > 0 {
            preprocess.push(prep_ms);
            latency.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    latency.sort_by(f64::total_cmp);
    preprocess.sort_by(f64::total_cmp);
    Ok(json!({"backend":"burn-flex", "samples":latency.len(),
    "p50_ms":latency[latency.len()/2], "p95_ms":latency[latency.len()*95/100],
    "preprocess_p50_ms":preprocess[preprocess.len()/2],
    "verification_time_ms":null,
    "process_peak_rss_kib":fs::read_to_string("/proc/self/status").ok().and_then(|status| {
        status.lines().find(|line| line.starts_with("VmHWM:"))
            .and_then(|line| line.split_whitespace().nth(1)).and_then(|value| value.parse::<u64>().ok())
    })}))
}

pub fn run(seed: u64, dataset: &Path, commit: &str, output: &Path) -> Result<()> {
    if ![7001, 7109, 7207, 7303].contains(&seed)
        || commit.len() != 40
        || !commit.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid registered seed or source SHA".into());
    }
    if dataset.join("answers/test.jsonl").exists() {
        return Err("sealed test answer file must be absent from the runner view".into());
    }
    fs::create_dir_all(output)?;
    let training: Vec<Case> = jsonl(&dataset.join("inputs/train.jsonl"))?;
    let validation: Vec<Case> = jsonl(&dataset.join("inputs/validation.jsonl"))?;
    let base_labels = targets(&training, jsonl(&dataset.join("answers/train.jsonl"))?)?;
    let validation_labels = targets(
        &validation,
        jsonl(&dataset.join("answers/validation.jsonl"))?,
    )?;
    assert_eq!((training.len(), validation.len()), (6144, 3072));
    let mut updates = serde_json::Map::new();
    let mut hashes = serde_json::Map::new();
    for path in [
        "inputs/train.jsonl",
        "inputs/validation.jsonl",
        "inputs/test.jsonl",
        "answers/train.jsonl",
        "answers/validation.jsonl",
    ] {
        hashes.insert(path.into(), hash(&fs::read(dataset.join(path))?).into());
    }
    let locked: Value = serde_json::from_str(include_str!(
        "../../../experiments/cycle7/BENCHMARK_LOCK.json"
    ))?;
    for (path, digest) in &hashes {
        if locked["files"][path] != *digest {
            return Err(format!("benchmark hash mismatch: {path}").into());
        }
    }
    let device = Device::cuda(0);
    let training_device = device.clone().autodiff();
    training_device.seed(seed);
    let initial = Ranker::new(&training_device);
    let parameter_count = initial.num_params();
    let initial_bytes = initial.checkpoint()?;
    let mut selections = Vec::new();
    for condition in CONDITIONS {
        let start = Instant::now();
        let dir = output.join(condition);
        fs::create_dir_all(&dir)?;
        let train_features: Vec<_> = training
            .iter()
            .map(|case| features(case, condition))
            .collect();
        let val_features: Vec<_> = validation
            .iter()
            .map(|case| features(case, condition))
            .collect();
        let mut labels = base_labels.clone();
        if condition == "shuffled_labels" {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x70_7707);
            for track in ["agent_evidence", "schema_map"] {
                let indices: Vec<_> = training
                    .iter()
                    .enumerate()
                    .filter(|(_, case)| case.track == track)
                    .map(|(index, _)| index)
                    .collect();
                let mut permuted: Vec<_> = indices.iter().map(|&i| labels[i]).collect();
                permuted.shuffle(&mut rng);
                for (i, label) in indices.into_iter().zip(permuted) {
                    labels[i] = label;
                }
            }
        }
        let mut model = Ranker::restore(&initial_bytes, &training_device)?;
        let mut optimizer = AdamConfig::new().init();
        let loss_fn = CrossEntropyLossConfig::new().init(&training_device);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xc700);
        let mut selected = None;
        let mut validation_candidates = Vec::new();
        let mut curve = BufWriter::new(File::create(dir.join("training_curve.csv"))?);
        writeln!(curve, "update,examples,loss,elapsed_seconds")?;
        for step in 1..=STEPS {
            let sample: Vec<_> = (0..BATCH)
                .map(|_| rng.gen_range(0..training.len()))
                .collect();
            let inputs = tensor(
                sample
                    .iter()
                    .flat_map(|&i| train_features[i].iter().copied())
                    .collect(),
                BATCH,
                &training_device,
            );
            let batch_labels = sample.iter().map(|&i| labels[i]).collect::<Vec<_>>();
            let labels_tensor = Tensor::<1, Int>::from_data(
                TensorData::new(batch_labels, [BATCH]),
                &training_device,
            );
            let loss = loss_fn.forward(model.forward(inputs), labels_tensor);
            let value = loss.clone().into_scalar::<f32>();
            assert!(value.is_finite(), "{condition} nonfinite loss at {step}");
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            assert_eq!(grads.len(), 4, "missing GPU gradients");
            model = optimizer.step(LR, model, grads);
            writeln!(
                curve,
                "{step},{},{value},{}",
                step * BATCH,
                start.elapsed().as_secs_f64()
            )?;
            if step % 1000 == 0 {
                curve.flush()?;
                println!("Cycle 7 seed={seed} condition={condition} update={step}/{STEPS} loss={value:.5}");
            }
            if [3000, 6000, 12000].contains(&step) {
                let bytes = model.checkpoint()?;
                assert_ne!(hash(&bytes), hash(&initial_bytes), "unchanged GPU weights");
                let eval = Ranker::restore(&bytes, &device)?;
                let predictions = predict(&eval, &validation, &val_features, &device);
                let correct = predictions
                    .iter()
                    .zip(&validation_labels)
                    .filter(|((class, _), label)| *class as i64 == **label)
                    .count();
                validation_candidates.push(json!({"update":step,"validation_exact_count":correct}));
                if selected.as_ref().is_none_or(|&(_, best)| correct > best) {
                    selected = Some((step, correct));
                    let probe = predict(
                        &model,
                        &validation[..BATCH],
                        &val_features[..BATCH],
                        &training_device,
                    );
                    let parity = probe.iter().zip(&predictions).all(|(a, b)| {
                        a.0 == b.0
                            && a.1.ranked_candidate_ids == b.1.ranked_candidate_ids
                            && (a.1.confidence - b.1.confidence).abs() < 1e-5
                    });
                    assert!(parity, "cold reload prediction parity failed");
                    fs::write(dir.join("model.safetensors"), &bytes)?;
                    let (cutoff, validation_precision_feasible) =
                        threshold(&validation, &predictions, &validation_labels);
                    predict_file(&dir.join("validation_predictions.jsonl"), &predictions)?;
                    write_json(
                        &dir.join("selection.json"),
                        &json!({"update":step, "validation_exact_count":correct,
                        "validation_confidence_threshold":cutoff, "validation_precision_feasible":validation_precision_feasible}),
                    )?;
                    write_json(
                        &dir.join("checkpoint.json"),
                        &json!({"sha256":hash(&bytes),"bytes":bytes.len(),
                        "cold_reload_parity":parity,"probe_cases":BATCH,"parameters":parameter_count}),
                    )?;
                }
            }
        }
        curve.flush()?;
        write_json(
            &dir.join("validation_candidates.json"),
            &json!(validation_candidates),
        )?;
        updates.insert(condition.into(), STEPS.into());
        selections.push(
            json!({"condition":condition,"optimizer_updates":STEPS,"training_examples":STEPS*BATCH,
            "training_wall_seconds": start.elapsed().as_secs_f64()}),
        );
    }
    // Checkpoint and threshold selection for every condition are complete.
    let test: Vec<Case> = jsonl(&dataset.join("inputs/test.jsonl"))?;
    assert_eq!(test.len(), 3072);
    for condition in CONDITIONS {
        let dir = output.join(condition);
        let bytes = fs::read(dir.join("model.safetensors"))?;
        let model = Ranker::restore(&bytes, &device)?;
        let test_features = test
            .iter()
            .map(|case| features(case, condition))
            .collect::<Vec<_>>();
        let predictions = predict(&model, &test, &test_features, &device);
        predict_file(&dir.join("test_predictions.jsonl"), &predictions)?;
        if condition == "hybrid" {
            for (split, cases) in [
                ("validation", validation.as_slice()),
                ("test", test.as_slice()),
            ] {
                let masked = cases
                    .iter()
                    .map(|case| {
                        let mut data = features(case, "hybrid");
                        for row in data.chunks_exact_mut(WIDTH) {
                            for field in [1, 2, 4, 5] {
                                row[field] = 0.0;
                            }
                        }
                        data
                    })
                    .collect::<Vec<_>>();
                let ablation = predict(&model, cases, &masked, &device);
                predict_file(
                    &dir.join(format!("{split}_identity_time_masked.jsonl")),
                    &ablation,
                )?;
            }
        }
        write_json(
            &dir.join("cpu_profile.json"),
            &cpu_profile(&bytes, &test, condition)?,
        )?;
    }
    write_json(
        &output.join("feature_manifest.json"),
        &json!({
            "architecture":"shared pointwise 512x1920x6 MLP, class logits by candidate, mean pooled NONE",
            "features":"track tag, FNV-1a binary alphabetic token buckets for text, explicit equality/unit/status/time-delta features for structure",
            "raw_case_candidate_and_family_id_features":false,
            "raw_trace_resource_scope_ids":false,
            "matched_pairwise_identity_only":true,
            "query_blind":"candidate text and track only",
            "shuffled_labels":"train labels permuted within track over fixed candidate slots",
            "batch":BATCH,"optimizer":"Adam","learning_rate":LR,
            "sampling":"ChaCha8 uniform train with replacement, shared ordering",
            "checkpoint_selection":"highest pooled validation exact count; earliest checkpoint wins ties",
            "threshold_selection":"maximum validation agent coverage at >=95% exact precision; 1 if no feasible threshold"
        }),
    )?;
    write_json(
        &output.join("run_manifest.json"),
        &json!({
            "seed":seed,"git_commit":commit,"backend":"burn-cuda","status":"completed",
            "model_parameters":parameter_count,"conditions": &CONDITIONS[..3],
            "negative_controls":&CONDITIONS[3..],"optimizer_updates_by_condition":updates,
            "training":selections,"dataset_hashes":hashes,"gradient_tensors_per_step":4
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle7_cpu_checkpoint_roundtrip() {
        let device = Device::flex();
        device.seed(7001);
        let model = Ranker::new(&device);
        let input = tensor(vec![0.125; WIDTH * 4], 1, &device);
        let bytes = model.checkpoint().unwrap();
        let restored = Ranker::restore(&bytes, &device).unwrap();
        assert_eq!(restored.num_params(), 996486);
        assert_eq!(restored.checkpoint().unwrap(), bytes);
        assert_eq!(
            model
                .forward(input.clone())
                .to_data_as::<f32>()
                .try_into_vec::<f32>()
                .unwrap(),
            restored
                .forward(input)
                .to_data_as::<f32>()
                .try_into_vec::<f32>()
                .unwrap(),
        );
    }

    #[test]
    fn cycle7_rank_mask_and_redaction() {
        let candidate = |id: &str| Candidate {
            id: id.into(),
            text: "trace=trace-000012 status=ok".into(),
            data: Value::Null,
        };
        let case = Case {
            case_id: "opaque".into(),
            track: "agent_evidence".into(),
            query: Value::Null,
            candidates: [
                candidate("c3"),
                candidate("c2"),
                candidate("c1"),
                candidate("c0"),
            ],
        };
        let mut logits = [0.0; 21];
        logits[2] = 100.0;
        logits[6] = 12.0;
        let (class, prediction) = decode(&case, &logits);
        assert_eq!(class, 6);
        assert_eq!(prediction.ranked_candidate_ids[0], "c2");
        assert_eq!(prediction.predicted_relation, "contradicts");
        assert_eq!(redact(&case.candidates[0].text), "status=ok");
    }
}
