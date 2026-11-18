//! Cycle 6: preregistered compositional transfer and sequential retention.
//! All neural computation, oracle generation, scoring and checkpointing is Rust.
//! See experiments/cycle6/PROTOCOL.md before changing any constant.

use crate::cycle5_model::Model;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Int, Tensor, TensorData};
use moonshot_core::TaskExample;
use moonshot_model::{ModelConfig, TinyTransformer};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, time::Instant};

const PARAMS: usize = 1_095_744;
const GRADIENT_TENSORS: usize = 38;
const TASKS: usize = 10;
const VALUES: usize = 8;
const IN_LEN: usize = 7;
const OUT_LEN: usize = 4;
const TRAIN_LEN: usize = IN_LEN + OUT_LEN - 1;
const BATCH: usize = 64;
const SEQUENTIAL_STEPS: usize = 400;
const COMPOSITION_STEPS: usize = 2_000;
const FEWSHOT_UPDATES: usize = 32;
const SUPPORT_SIZE: usize = 16;
const LEARNING_RATE: f64 = 0.001;
const TRAIN_SET: usize = 384;
const VALIDATION_SET: usize = 64;
const TEST_SET: usize = 64;
const PAIR_TRAIN: usize = 48;
const PAIR_VALIDATION: usize = 20;
const PAIR_TEST: usize = 20;
const PAIR_SAMPLES: usize = 64;
const DEV_SEED: u64 = 6001;
const EVAL_SEEDS: [u64; 3] = [6109, 6203, 6301];
const BOS: usize = 1;
const EOS: usize = 2;
const SEP: usize = 3;
const SYMBOL: usize = 4;
const OP: usize = 20;
const NO_OP: usize = 30;

type State = [usize; 3];

#[derive(Clone, Serialize)]
struct Split {
    train: Vec<TaskExample>,
    validation: Vec<TaskExample>,
    test: Vec<TaskExample>,
}

#[derive(Clone, Serialize)]
struct PairGroup {
    a: usize,
    b: usize,
    role: String,
    training: Vec<TaskExample>,
    evaluation: Vec<TaskExample>,
    support: Vec<TaskExample>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Current,
    Uniform,
    Feedback,
    Mixed,
}

#[derive(Serialize)]
struct ReplayProbe {
    step: usize,
    train_exact_match: Vec<f64>,
    task_sampling_weights: Vec<f64>,
}

#[derive(Serialize)]
struct TrainTrace {
    steps: usize,
    batch: usize,
    learning_rate: f64,
    losses: Vec<(usize, f32)>,
    probe_forward_examples: usize,
    replay_probes: Vec<ReplayProbe>,
    sampled_previous_task_counts: Vec<usize>,
    wall_seconds: f64,
    gradients_per_step: usize,
    weights_changed: bool,
}

#[derive(Serialize)]
struct Eval {
    exact_match: f64,
    predictions: Vec<Vec<usize>>,
}

fn state_op(op: usize, [a, b, c]: State) -> State {
    match op {
        0 => [b, c, a],
        1 => [c, a, b],
        2 => [(a + 1) % 8, (b + 1) % 8, (c + 1) % 8],
        3 => [(a + 2) % 8, b, c],
        4 => [c, b, a],
        5 => [a ^ b, b ^ c, c ^ a],
        6 => [(a + b) % 8, (b + c) % 8, (c + a) % 8],
        7 => {
            if a % 2 == 0 {
                [a, c, b]
            } else {
                [a, b, c]
            }
        }
        8 => [a, (b + a) % 8, (c + a) % 8],
        9 => [a, a ^ b, b ^ c],
        _ => panic!("unregistered operator {op}"),
    }
}

fn all_states() -> Vec<State> {
    (0..VALUES)
        .flat_map(|a| (0..VALUES).flat_map(move |b| (0..VALUES).map(move |c| [a, b, c])))
        .collect()
}

fn example(a: usize, b: Option<usize>, input: State) -> TaskExample {
    let after_a = state_op(a, input);
    let out = b.map_or(after_a, |second| state_op(second, after_a));
    TaskExample {
        input_tokens: vec![
            BOS,
            OP + a,
            b.map_or(NO_OP, |op| OP + op),
            SYMBOL + input[0],
            SYMBOL + input[1],
            SYMBOL + input[2],
            SEP,
        ],
        target_tokens: vec![SYMBOL + out[0], SYMBOL + out[1], SYMBOL + out[2], EOS],
    }
}

fn generate_primitives() -> Vec<Split> {
    (0..TASKS)
        .map(|op| {
            let mut states = all_states();
            states.shuffle(&mut ChaCha8Rng::seed_from_u64(606600 + op as u64));
            let cases: Vec<_> = states
                .into_iter()
                .map(|state| example(op, None, state))
                .collect();
            Split {
                train: cases[..TRAIN_SET].to_vec(),
                validation: cases[TRAIN_SET..TRAIN_SET + VALIDATION_SET].to_vec(),
                test: cases[TRAIN_SET + VALIDATION_SET..].to_vec(),
            }
        })
        .collect()
}

fn redundant_pair(a: usize, b: usize, states: &[State]) -> bool {
    (0..=TASKS).any(|candidate| {
        states.iter().all(|&state| {
            let composed = state_op(b, state_op(a, state));
            if candidate == TASKS {
                composed == state
            } else {
                composed == state_op(candidate, state)
            }
        })
    })
}

fn generate_pairs() -> Vec<PairGroup> {
    let states = all_states();
    let mut pairs: Vec<_> = (0..TASKS)
        .flat_map(|a| (0..TASKS).filter(move |&b| a != b).map(move |b| (a, b)))
        .filter(|&(a, b)| !redundant_pair(a, b, &states))
        .collect();
    assert_eq!(pairs.len(), PAIR_TRAIN + PAIR_VALIDATION + PAIR_TEST);
    pairs.shuffle(&mut ChaCha8Rng::seed_from_u64(606688));
    pairs
        .into_iter()
        .enumerate()
        .map(|(index, (a, b))| {
            let role = if index < PAIR_TRAIN {
                "train"
            } else if index < PAIR_TRAIN + PAIR_VALIDATION {
                "validation"
            } else {
                "test"
            };
            let mut states = states.clone();
            states.shuffle(&mut ChaCha8Rng::seed_from_u64(
                600_000 + (a * TASKS + b) as u64,
            ));
            let training = if role == "train" {
                states[..PAIR_SAMPLES]
                    .iter()
                    .map(|&state| example(a, Some(b), state))
                    .collect()
            } else {
                Vec::new()
            };
            let evaluation = if role != "train" {
                states[PAIR_SAMPLES..2 * PAIR_SAMPLES]
                    .iter()
                    .map(|&state| example(a, Some(b), state))
                    .collect()
            } else {
                Vec::new()
            };
            let support = if role == "test" {
                states[2 * PAIR_SAMPLES..2 * PAIR_SAMPLES + SUPPORT_SIZE]
                    .iter()
                    .map(|&state| example(a, Some(b), state))
                    .collect()
            } else {
                Vec::new()
            };
            PairGroup {
                a,
                b,
                role: role.into(),
                training,
                evaluation,
                support,
            }
        })
        .collect()
}

fn validate_datasets(primitives: &[Split], pairs: &[PairGroup]) {
    use std::collections::HashSet;
    assert_eq!(primitives.len(), TASKS);
    for (op, split) in primitives.iter().enumerate() {
        assert_eq!(
            (split.train.len(), split.validation.len(), split.test.len()),
            (TRAIN_SET, VALIDATION_SET, TEST_SET)
        );
        let examples = split
            .train
            .iter()
            .chain(&split.validation)
            .chain(&split.test);
        let mut unique = HashSet::new();
        for ex in examples {
            assert!(
                unique.insert(ex.input_tokens.clone()),
                "task {op} split overlap"
            );
            assert_eq!(ex.input_tokens.len(), IN_LEN);
            assert_eq!(ex.target_tokens.len(), OUT_LEN);
            assert_eq!(ex.target_tokens[OUT_LEN - 1], EOS);
            assert_eq!(ex.input_tokens[1], OP + op);
            assert_eq!(ex.input_tokens[2], NO_OP);
        }
        assert_eq!(unique.len(), VALUES.pow(3));
    }
    assert_eq!(pairs.len(), 88);
    let mut pair_ids = HashSet::new();
    let mut counts = [0; 3];
    for pair in pairs {
        assert!(pair_ids.insert((pair.a, pair.b)));
        match pair.role.as_str() {
            "train" => {
                counts[0] += 1;
                assert_eq!(pair.training.len(), PAIR_SAMPLES);
            }
            "validation" => {
                counts[1] += 1;
                assert_eq!(pair.evaluation.len(), PAIR_SAMPLES);
            }
            "test" => {
                counts[2] += 1;
                assert_eq!(pair.evaluation.len(), PAIR_SAMPLES);
                assert_eq!(pair.support.len(), SUPPORT_SIZE);
                let test: HashSet<_> = pair
                    .evaluation
                    .iter()
                    .map(|x| x.input_tokens.clone())
                    .collect();
                assert!(pair.support.iter().all(|x| !test.contains(&x.input_tokens)));
            }
            _ => panic!("unexpected pair role"),
        }
    }
    assert_eq!(counts, [PAIR_TRAIN, PAIR_VALIDATION, PAIR_TEST]);
}

fn token_tensor(examples: &[TaskExample], device: &Device, labels: bool) -> Tensor<2, Int> {
    let width = if labels { OUT_LEN } else { TRAIN_LEN };
    let tokens: Vec<i64> = examples
        .iter()
        .flat_map(|example| {
            if labels {
                example.target_tokens.clone()
            } else {
                example
                    .input_tokens
                    .iter()
                    .chain(&example.target_tokens[..OUT_LEN - 1])
                    .copied()
                    .collect()
            }
        })
        .map(|v| v as i64)
        .collect();
    Tensor::from_data(TensorData::new(tokens, [examples.len(), width]), device)
}

fn evaluate(
    model: &Model,
    examples: &[TaskExample],
    config: &ModelConfig,
    device: &Device,
) -> Eval {
    if examples.is_empty() {
        return Eval {
            exact_match: 0.0,
            predictions: Vec::new(),
        };
    }
    let batch = examples.len();
    let mut sequences: Vec<_> = examples.iter().map(|ex| ex.input_tokens.clone()).collect();
    for len in IN_LEN..IN_LEN + OUT_LEN {
        let data: Vec<i64> = sequences
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as i64))
            .collect();
        let input = Tensor::<2, Int>::from_data(TensorData::new(data, [batch, len]), device);
        let prediction = model
            .forward(input, config.n_heads)
            .slice([0..batch, len - 1..len, 0..config.vocab_size])
            .argmax(2)
            .to_data_as::<i64>()
            .try_into_vec::<i64>()
            .expect("CUDA inference data");
        for (sequence, token) in sequences.iter_mut().zip(prediction) {
            sequence.push(token as usize);
        }
    }
    let predictions: Vec<_> = sequences
        .into_iter()
        .map(|s| s[IN_LEN..].to_vec())
        .collect();
    let exact = predictions
        .iter()
        .zip(examples)
        .filter(|(prediction, item)| **prediction == item.target_tokens)
        .count();
    Eval {
        exact_match: exact as f64 / batch as f64,
        predictions,
    }
}

fn sample(rng: &mut ChaCha8Rng, values: &[TaskExample], count: usize, dst: &mut Vec<TaskExample>) {
    assert!(!values.is_empty());
    for _ in 0..count {
        dst.push(values[rng.gen_range(0..values.len())].clone());
    }
}

fn weighted_choice(rng: &mut ChaCha8Rng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    assert!(total.is_finite() && total > 0.0);
    let mut pick = rng.gen::<f64>() * total;
    for (index, &weight) in weights.iter().enumerate() {
        pick -= weight;
        if pick < 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

/// Rust-native Burn training. Optimizer resets at each task or adaptation boundary.
fn train(
    mut model: Model,
    current: &[TaskExample],
    previous: &[Vec<TaskExample>],
    second: Option<&[TaskExample]>,
    mode: Strategy,
    steps: usize,
    batch_size: usize,
    seed: u64,
    config: &ModelConfig,
    device: &Device,
) -> (Model, TrainTrace) {
    let timer = Instant::now();
    let initial = model.parameters();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut optimizer = AdamConfig::new().init();
    let cross_entropy = CrossEntropyLossConfig::new().init(device);
    let mut losses = Vec::new();
    let mut probes = 0usize;
    let mut weights = vec![1.0f64; previous.len()];
    let mut replay_probes = Vec::new();
    let mut sampled_previous_task_counts = vec![0usize; previous.len()];
    assert!(!current.is_empty());
    assert!(mode != Strategy::Mixed || second.is_some());
    for step in 0..steps {
        if !previous.is_empty()
            && matches!(mode, Strategy::Uniform | Strategy::Feedback)
            && step % 50 == 0
        {
            let mut scores = Vec::with_capacity(previous.len());
            for (index, task) in previous.iter().enumerate() {
                let probe = &task[..task.len().min(32)];
                let score = evaluate(&model, probe, config, device).exact_match;
                weights[index] = 1.0 + 4.0 * (1.0 - score);
                scores.push(score);
                probes += probe.len();
            }
            replay_probes.push(ReplayProbe {
                step,
                train_exact_match: scores,
                task_sampling_weights: if mode == Strategy::Uniform {
                    vec![1.0; previous.len()]
                } else {
                    weights.clone()
                },
            });
        }
        let mut batch = Vec::with_capacity(batch_size);
        if mode == Strategy::Mixed {
            sample(&mut rng, current, batch_size / 2, &mut batch);
            sample(
                &mut rng,
                second.expect("mixed second pool"),
                batch_size - batch.len(),
                &mut batch,
            );
        } else if previous.is_empty() || mode == Strategy::Current {
            sample(&mut rng, current, batch_size, &mut batch);
        } else {
            sample(&mut rng, current, batch_size / 2, &mut batch);
            for _ in batch.len()..batch_size {
                let task = if mode == Strategy::Feedback {
                    weighted_choice(&mut rng, &weights)
                } else {
                    rng.gen_range(0..previous.len())
                };
                sampled_previous_task_counts[task] += 1;
                sample(&mut rng, &previous[task], 1, &mut batch);
            }
        }
        let input = token_tensor(&batch, device, false);
        let labels = token_tensor(&batch, device, true).reshape([batch_size * OUT_LEN]);
        let logits = model
            .forward(input, config.n_heads)
            .slice([0..batch_size, IN_LEN - 1..TRAIN_LEN, 0..config.vocab_size])
            .reshape([batch_size * OUT_LEN, config.vocab_size]);
        let loss = cross_entropy.forward(logits, labels);
        let scalar: f32 = loss.clone().into_scalar();
        assert!(scalar.is_finite(), "Cycle 6 nonfinite loss at step {step}");
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        assert_eq!(
            grads.len(),
            GRADIENT_TENSORS,
            "missing gradients at step {step}"
        );
        model = optimizer.step(LEARNING_RATE, model, grads);
        if step == 0 || (step + 1) % 25 == 0 || step + 1 == steps {
            losses.push((step + 1, scalar));
        }
        if (step + 1) % 100 == 0 || step + 1 == steps {
            println!("cycle6 steps={}/{} loss={scalar:.6}", step + 1, steps);
        }
    }
    let after = model.parameters();
    assert_eq!(after.len(), PARAMS);
    assert!(after.iter().all(|v| v.is_finite()));
    let changed = initial != after;
    assert!(changed, "Cycle 6 optimizer did not update weights");
    (
        model,
        TrainTrace {
            steps,
            batch: batch_size,
            learning_rate: LEARNING_RATE,
            losses,
            probe_forward_examples: probes,
            replay_probes,
            sampled_previous_task_counts,
            wall_seconds: timer.elapsed().as_secs_f64(),
            gradients_per_step: GRADIENT_TENSORS,
            weights_changed: changed,
        },
    )
}

fn save_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn restore_for_inference(model: &Model, config: &ModelConfig, seed: u64, device: &Device) -> Model {
    let mut native = TinyTransformer::new(config.clone(), seed);
    native.load_parameters(&model.parameters());
    Model::from_moonshot(&native, device)
}

fn checkpoint(
    model: &Model,
    config: &ModelConfig,
    seed: u64,
    dir: &Path,
    probe: &[TaskExample],
    train_device: &Device,
    inference_device: &Device,
) -> Result<Value, Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    let mut native = TinyTransformer::new(config.clone(), seed);
    native.load_parameters(&model.parameters());
    let bytes = native.to_safetensors()?;
    fs::write(dir.join("model.safetensors"), &bytes)?;
    let restored = TinyTransformer::from_safetensors(&fs::read(dir.join("model.safetensors"))?)?;
    assert_eq!(restored.extract_parameters(), native.extract_parameters());
    let original_predictions = evaluate(model, probe, config, train_device).predictions;
    let cold_model = Model::from_moonshot(&restored, inference_device);
    let cold_predictions = evaluate(&cold_model, probe, config, inference_device).predictions;
    assert_eq!(
        original_predictions, cold_predictions,
        "cold checkpoint inference mismatch"
    );
    Ok(
        json!({"file": "model.safetensors", "sha256": digest(&bytes),
              "bytes": bytes.len(), "parameters": PARAMS,
              "cold_reload_predicted_same": true, "probe_count": probe.len()}),
    )
}

fn save_predictions(
    dir: &Path,
    name: &str,
    tasks: &[(String, &[TaskExample], &Eval)],
) -> Result<(), Box<dyn Error>> {
    let mut records = Vec::new();
    for (task, examples, evaluation) in tasks {
        assert_eq!(examples.len(), evaluation.predictions.len());
        for (ex, predicted) in examples.iter().zip(&evaluation.predictions) {
            records.push(json!({
                "task": task, "input": &ex.input_tokens, "target": &ex.target_tokens,
                "predicted": predicted, "correct": *predicted == ex.target_tokens,
            }));
        }
    }
    save_json(&dir.join(name), &records)
}

fn average(numbers: &[f64]) -> f64 {
    if numbers.is_empty() {
        0.0
    } else {
        numbers.iter().sum::<f64>() / numbers.len() as f64
    }
}

fn run_sequential(
    seed: u64,
    dir: &Path,
    primitives: &[Split],
    base: &TinyTransformer,
    config: &ModelConfig,
    train_device: &Device,
    inference_device: &Device,
) -> Result<Value, Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    println!("[C6] seed={seed} isolated-task capability ceiling");
    let mut isolated_scores = Vec::new();
    let mut isolated_results = Vec::new();
    let mut isolated_predictions = Vec::new();
    for (task_id, task) in primitives.iter().enumerate() {
        let initial = Model::from_moonshot(base, train_device);
        let (model, trace) = train(
            initial,
            &task.train,
            &[],
            None,
            Strategy::Current,
            SEQUENTIAL_STEPS,
            BATCH,
            seed ^ ((task_id as u64 + 1) << 32),
            config,
            train_device,
        );
        let inference = restore_for_inference(&model, config, seed, inference_device);
        let score = evaluate(&inference, &task.test, config, inference_device);
        isolated_scores.push(score.exact_match);
        let checkpoint = checkpoint(
            &model,
            config,
            seed,
            &dir.join(format!("isolated_{task_id}")),
            &task.test[..8],
            train_device,
            inference_device,
        )?;
        isolated_predictions.push((task_id, score.predictions));
        isolated_results.push(json!({
            "task": task_id, "test_accuracy": score.exact_match,
            "training": trace, "checkpoint": checkpoint,
        }));
        println!(
            "[C6] seed={seed} isolated task={task_id} heldout={:.3}",
            score.exact_match
        );
        save_json(&dir.join("isolated_progress.json"), &isolated_results)?;
    }
    let acquired: Vec<_> = isolated_scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= 0.80)
        .map(|(task, _)| task)
        .collect();
    let isolated_records: Vec<_> = isolated_predictions.into_iter().map(|(task, predictions)| {
        json!({"task": task, "input_target": &primitives[task].test, "predictions": predictions})
    }).collect();
    save_json(&dir.join("isolated_predictions.json"), &isolated_records)?;

    let mut conditions = serde_json::Map::new();
    let mut final_scores = std::collections::BTreeMap::<&'static str, Vec<f64>>::new();
    for (name, mode) in [
        ("shared", Strategy::Current),
        ("uniform", Strategy::Uniform),
        ("feedback", Strategy::Feedback),
    ] {
        println!("[C6] seed={seed} sequential condition={name}");
        let mut model = Model::from_moonshot(base, train_device);
        let mut matrix: Vec<Vec<Option<f64>>> = Vec::new();
        let mut trajectory = Vec::new();
        let mut training = Vec::new();
        let mut previous: Vec<Vec<TaskExample>> = Vec::new();
        for (task_id, task) in primitives.iter().enumerate() {
            let (updated, trace) = train(
                model,
                &task.train,
                &previous,
                None,
                mode,
                SEQUENTIAL_STEPS,
                BATCH,
                seed ^ ((task_id as u64 + 1) << 24) ^ 0xA11C_6000,
                config,
                train_device,
            );
            model = updated;
            let inference = restore_for_inference(&model, config, seed, inference_device);
            let mut row = vec![None; TASKS];
            for old_task in 0..=task_id {
                let checked = evaluate(
                    &inference,
                    &primitives[old_task].test,
                    config,
                    inference_device,
                );
                row[old_task] = Some(checked.exact_match);
                trajectory.push(json!({
                    "generation": task_id + 1, "task": old_task,
                    "examples": &primitives[old_task].test,
                    "predictions": checked.predictions,
                    "exact_match": checked.exact_match,
                }));
            }
            println!(
                "[C6] seed={seed} {name} generation={} current={:.3} prev_mean={:.3}",
                task_id + 1,
                row[task_id].unwrap_or(0.0),
                average(&row[..task_id].iter().flatten().copied().collect::<Vec<_>>())
            );
            matrix.push(row);
            training.push(trace);
            previous.push(task.train.clone());
            save_json(
                &dir.join(format!("{name}_progress.json")),
                &json!({
                    "matrix": &matrix, "training": &training,
                }),
            )?;
            save_json(
                &dir.join(format!("{name}_trajectory_predictions.json")),
                &trajectory,
            )?;
        }
        let inference = restore_for_inference(&model, config, seed, inference_device);
        let final_test: Vec<_> = primitives
            .iter()
            .map(|task| evaluate(&inference, &task.test, config, inference_device))
            .collect();
        let final_acc: Vec<f64> = final_test.iter().map(|item| item.exact_match).collect();
        let predictions: Vec<_> = final_test
            .iter()
            .enumerate()
            .map(|(i, eval)| {
                (
                    format!("primitive_{i}"),
                    primitives[i].test.as_slice(),
                    eval,
                )
            })
            .collect();
        save_predictions(dir, &format!("{name}_final_predictions.json"), &predictions)?;
        let checkpoint = checkpoint(
            &model,
            config,
            seed,
            &dir.join(format!("{name}_final")),
            &primitives[0].test[..8],
            train_device,
            inference_device,
        )?;
        let acquired_accuracy = average(
            &acquired
                .iter()
                .map(|&task| final_acc[task])
                .collect::<Vec<_>>(),
        );
        conditions.insert(
            name.into(),
            json!({
                "retention_matrix": matrix, "training": training,
                "final_test_accuracy": &final_acc,
                "all_task_mean": average(&final_acc),
                "acquired_task_mean": acquired_accuracy,
                "last_task_accuracy": final_acc[TASKS - 1],
                "checkpoint": checkpoint,
            }),
        );
        final_scores.insert(name, final_acc);
    }

    let past_acquired: Vec<_> = acquired
        .iter()
        .copied()
        .filter(|&i| i + 1 < TASKS)
        .collect();
    let acquired_mean = |name: &str| -> f64 {
        average(
            &past_acquired
                .iter()
                .map(|&i| final_scores[name][i])
                .collect::<Vec<_>>(),
        )
    };
    let feedback = acquired_mean("feedback");
    let uniform = acquired_mean("uniform");
    let shared = acquired_mean("shared");
    let acquisition_gate = acquired.len() >= 7;
    let feedback_condition = &conditions["feedback"]["retention_matrix"];
    let feedback_acquired_at_birth = acquired
        .iter()
        .all(|&task| feedback_condition[task][task].as_f64().unwrap_or(0.0) >= 0.80);
    let retention_gate = acquisition_gate
        && feedback >= 0.70
        && feedback - uniform >= 0.10
        && feedback - shared >= 0.20
        && feedback_acquired_at_birth
        && final_scores["feedback"][TASKS - 1] >= 0.80;
    let output = json!({
        "seed": seed, "isolated": isolated_results,
        "isolated_scores": isolated_scores, "acquired_task_ids": acquired,
        "past_acquired_task_ids": past_acquired,
        "acquisition_gate": acquisition_gate,
        "conditions": conditions,
        "retention_comparison": {
            "feedback_acquired_mean": feedback,
            "uniform_acquired_mean": uniform,
            "shared_acquired_mean": shared,
            "feedback_minus_uniform": feedback - uniform,
            "feedback_minus_shared": feedback - shared,
            "feedback_acquired_at_birth": feedback_acquired_at_birth,
        },
        "retention_gate": retention_gate,
    });
    save_json(&dir.join("sequential_result.json"), &output)?;
    Ok(output)
}

fn pair_accuracy(
    model: &Model,
    pairs: &[&PairGroup],
    config: &ModelConfig,
    device: &Device,
) -> Vec<Eval> {
    pairs
        .iter()
        .map(|group| evaluate(model, &group.evaluation, config, device))
        .collect()
}

fn simple_pair_baseline(group: &PairGroup, first_only: bool) -> f64 {
    let correct = group
        .evaluation
        .iter()
        .filter(|example| {
            let original = [
                example.input_tokens[3] - SYMBOL,
                example.input_tokens[4] - SYMBOL,
                example.input_tokens[5] - SYMBOL,
            ];
            let output = if first_only {
                state_op(group.a, original)
            } else {
                original
            };
            example.target_tokens
                == vec![
                    SYMBOL + output[0],
                    SYMBOL + output[1],
                    SYMBOL + output[2],
                    EOS,
                ]
        })
        .count();
    correct as f64 / group.evaluation.len() as f64
}

fn run_composition(
    seed: u64,
    dir: &Path,
    primitives: &[Split],
    pairs: &[PairGroup],
    base: &TinyTransformer,
    config: &ModelConfig,
    train_device: &Device,
    inference_device: &Device,
) -> Result<Value, Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    let primitive_train: Vec<_> = primitives
        .iter()
        .flat_map(|x| x.train.iter().cloned())
        .collect();
    let pair_train: Vec<_> = pairs
        .iter()
        .filter(|p| p.role == "train")
        .flat_map(|p| p.training.iter().cloned())
        .collect();
    let pair_validation: Vec<_> = pairs.iter().filter(|p| p.role == "validation").collect();
    let pair_test: Vec<_> = pairs.iter().filter(|p| p.role == "test").collect();
    assert_eq!(pair_train.len(), PAIR_TRAIN * PAIR_SAMPLES);
    assert_eq!(pair_validation.len(), PAIR_VALIDATION);
    assert_eq!(pair_test.len(), PAIR_TEST);

    let mut weights_by_condition = std::collections::BTreeMap::<String, Vec<f32>>::new();
    let mut pretrained = serde_json::Map::new();
    let mut final_scores = std::collections::BTreeMap::<String, f64>::new();
    let mut source_skills = std::collections::BTreeMap::<String, usize>::new();
    for (name, mode) in [
        ("primitives_only", Strategy::Current),
        ("pairs_and_primitives", Strategy::Mixed),
    ] {
        println!("[C6] seed={seed} composition pretraining={name}");
        let initial = Model::from_moonshot(base, train_device);
        let (model, trace) = train(
            initial,
            &primitive_train,
            &[],
            if mode == Strategy::Mixed {
                Some(pair_train.as_slice())
            } else {
                None
            },
            mode,
            COMPOSITION_STEPS,
            BATCH,
            seed ^ 0xCAFE_0000,
            config,
            train_device,
        );
        let inference = restore_for_inference(&model, config, seed, inference_device);
        let heldout = pair_accuracy(&inference, &pair_test, config, inference_device);
        let validation = pair_accuracy(&inference, &pair_validation, config, inference_device);
        let test_scores: Vec<_> = heldout.iter().map(|e| e.exact_match).collect();
        let validation_scores: Vec<_> = validation.iter().map(|e| e.exact_match).collect();
        let source_evals: Vec<_> = primitives
            .iter()
            .map(|task| evaluate(&inference, &task.test, config, inference_device))
            .collect();
        let primitive_scores: Vec<_> = source_evals.iter().map(|e| e.exact_match).collect();
        let source_predictions: Vec<_> = source_evals
            .into_iter()
            .enumerate()
            .map(|(i, tested)| {
                json!({"task": i, "test": &primitives[i].test,
                   "predictions": tested.predictions, "exact_match": tested.exact_match})
            })
            .collect();
        save_json(
            &dir.join(format!("{name}_primitive_predictions.json")),
            &source_predictions,
        )?;
        let acquired = primitive_scores
            .iter()
            .filter(|&&score| score >= 0.8)
            .count();
        let checkpoint_info = checkpoint(
            &model,
            config,
            seed,
            &dir.join(name),
            &primitives[0].test[..8],
            train_device,
            inference_device,
        )?;
        let predictions: Vec<_> = pair_test
            .iter()
            .zip(&heldout)
            .map(|(pair, result)| {
                json!({"pair": [pair.a, pair.b], "test": &pair.evaluation,
                   "predictions": &result.predictions, "exact_match": result.exact_match})
            })
            .collect();
        save_json(
            &dir.join(format!("{name}_heldout_predictions.json")),
            &predictions,
        )?;
        final_scores.insert(name.into(), average(&test_scores));
        source_skills.insert(name.into(), acquired);
        pretrained.insert(
            name.into(),
            json!({
                "train": trace, "validation_pair_scores": &validation_scores,
                "test_pair_scores": &test_scores,
                "heldout_pair_mean": average(&test_scores),
                "primitive_test_scores": &primitive_scores,
                "primitive_skills_acquired": acquired,
                "checkpoint": checkpoint_info,
            }),
        );
        weights_by_condition.insert(name.into(), model.parameters());
    }

    let copy_scores: Vec<_> = pair_test
        .iter()
        .map(|p| simple_pair_baseline(p, false))
        .collect();
    let first_scores: Vec<_> = pair_test
        .iter()
        .map(|p| simple_pair_baseline(p, true))
        .collect();
    let copy_accuracy = average(&copy_scores);
    let first_accuracy = average(&first_scores);

    let mut fewshot_scores = serde_json::Map::new();
    let mut fewshot_means = std::collections::BTreeMap::<String, f64>::new();
    for name in ["random_init", "primitives_only", "pairs_and_primitives"] {
        println!("[C6] seed={seed} 16-shot condition={name}");
        let mut records = Vec::new();
        let mut results = Vec::new();
        for pair in &pair_test {
            let mut native = TinyTransformer::new(config.clone(), seed);
            if let Some(weights) = weights_by_condition.get(name) {
                native.load_parameters(weights);
            }
            let model = Model::from_moonshot(&native, train_device);
            let (adapted, trace) = train(
                model,
                &pair.support,
                &[],
                None,
                Strategy::Current,
                FEWSHOT_UPDATES,
                SUPPORT_SIZE,
                seed ^ 0xABCD_0000 ^ ((pair.a * TASKS + pair.b) as u64),
                config,
                train_device,
            );
            let inference = restore_for_inference(&adapted, config, seed, inference_device);
            let evaluated = evaluate(&inference, &pair.evaluation, config, inference_device);
            records.push(json!({"pair": [pair.a, pair.b], "test": &pair.evaluation,
                "support": &pair.support, "predictions": &evaluated.predictions,
                "exact_match": evaluated.exact_match,
            }));
            results.push(json!({"pair": [pair.a, pair.b],
                "test_exact_match": evaluated.exact_match, "training": trace,
            }));
        }
        let scores: Vec<f64> = results
            .iter()
            .map(|record| {
                record["test_exact_match"]
                    .as_f64()
                    .expect("finite fewshot accuracy")
            })
            .collect();
        fewshot_means.insert(name.into(), average(&scores));
        fewshot_scores.insert(
            name.into(),
            json!({
                "per_pair": results, "mean_exact_match": average(&scores),
            }),
        );
        save_json(
            &dir.join(format!("fewshot_{name}_predictions.json")),
            &records,
        )?;
        save_json(&dir.join("fewshot_progress.json"), &fewshot_scores)?;
    }

    let joint = final_scores["pairs_and_primitives"];
    let primitive = final_scores["primitives_only"];
    let joint_fewshot = fewshot_means["pairs_and_primitives"];
    let primitive_fewshot = fewshot_means["primitives_only"];
    let source_acquired = source_skills["pairs_and_primitives"];
    let gate = source_acquired >= 7
        && joint >= 0.50
        && joint - primitive >= 0.20
        && joint - copy_accuracy >= 0.20
        && joint - first_accuracy >= 0.20
        && joint_fewshot - primitive_fewshot >= 0.10;
    let output = json!({
        "seed": seed,
        "pretraining": pretrained,
        "no_learning_baselines": {
            "identity_copy_per_pair": copy_scores,
            "first_operator_per_pair": first_scores,
            "identity_copy_mean": copy_accuracy,
            "first_operator_mean": first_accuracy,
        },
        "fewshot": fewshot_scores,
        "comparison": {
            "joint_zero_shot": joint, "primitive_zero_shot": primitive,
            "joint_minus_primitive": joint - primitive,
            "joint_16shot": joint_fewshot,
            "primitive_16shot": primitive_fewshot,
            "joint_16shot_minus_primitive": joint_fewshot - primitive_fewshot,
            "joint_source_primitives_acquired": source_acquired,
        },
        "composition_gate": gate,
    });
    save_json(&dir.join("composition_result.json"), &output)?;
    Ok(output)
}

pub fn run(seed: u64, dir: &Path, commit: &str) -> Result<(), Box<dyn Error>> {
    if !EVAL_SEEDS.contains(&seed) && seed != DEV_SEED {
        return Err(format!("seed {seed} is not preregistered").into());
    }
    if commit.len() != 40 || !commit.bytes().all(|x| x.is_ascii_hexdigit()) {
        return Err("requires a 40-character pinned Git SHA".into());
    }
    fs::create_dir_all(dir)?;
    let timer = Instant::now();
    let config = ModelConfig::new_1m();
    let gpu = Device::cuda(0);
    let train_device = gpu.clone().autodiff();
    let base = TinyTransformer::new(config.clone(), seed);
    assert_eq!(
        base.extract_parameters().len(),
        PARAMS,
        "wrong Moonshot model size"
    );
    let primitives = generate_primitives();
    let pairs = generate_pairs();
    validate_datasets(&primitives, &pairs);
    save_json(&dir.join("primitive_splits.json"), &primitives)?;
    save_json(&dir.join("operator_pair_splits.json"), &pairs)?;
    let dataset_sha256 = json!({
        "primitive_splits": digest(&fs::read(dir.join("primitive_splits.json"))?),
        "operator_pair_splits": digest(&fs::read(dir.join("operator_pair_splits.json"))?),
    });
    let mut status = json!({
        "cycle": 6, "status": "running", "seed": seed,
        "git_commit": commit, "backend": "burn-cuda", "model_parameters": PARAMS,
        "dataset_sha256": &dataset_sha256, "trainer": "rust-burn-cuda",
        "domain": "ten-symbolic-state-operators-with-heldout-ordered-compositions",
        "train_steps_per_primitive": SEQUENTIAL_STEPS,
        "composition_train_steps": COMPOSITION_STEPS,
        "fewshot_adaptation_steps": FEWSHOT_UPDATES,
    });
    save_json(&dir.join("cycle6_run.json"), &status)?;
    let sequential = run_sequential(
        seed,
        &dir.join("sequential"),
        &primitives,
        &base,
        &config,
        &train_device,
        &gpu,
    )?;
    status["sequential_completed"] = true.into();
    save_json(&dir.join("cycle6_run.json"), &status)?;
    let composition = run_composition(
        seed,
        &dir.join("composition"),
        &primitives,
        &pairs,
        &base,
        &config,
        &train_device,
        &gpu,
    )?;
    status["composition_completed"] = true.into();
    let success = sequential["acquisition_gate"].as_bool() == Some(true)
        && sequential["retention_gate"].as_bool() == Some(true)
        && composition["composition_gate"].as_bool() == Some(true);
    status["status"] = "completed".into();
    status["scientific_leap_gate"] = success.into();
    status["total_wall_seconds"] = timer.elapsed().as_secs_f64().into();
    save_json(&dir.join("cycle6_run.json"), &status)?;
    println!("[C6] seed={seed} finished scientific_leap_gate={success}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn frozen_oracle_and_composition_partitions_are_nontrivial() {
        let primitives = generate_primitives();
        let pairs = generate_pairs();
        validate_datasets(&primitives, &pairs);
        assert_eq!(pairs.iter().filter(|p| p.role == "test").count(), 20);
        let train: HashSet<_> = pairs
            .iter()
            .filter(|p| p.role == "train")
            .map(|p| (p.a, p.b))
            .collect();
        for pair in pairs.iter().filter(|p| p.role == "test") {
            assert!(!train.contains(&(pair.a, pair.b)));
            assert!(!redundant_pair(pair.a, pair.b, &all_states()));
        }
        assert_eq!(
            generate_primitives()[0].test[0].target_tokens.len(),
            OUT_LEN
        );
    }

    #[test]
    fn cheap_copy_control_is_not_an_oracle_for_complex_composition() {
        let pairs = generate_pairs();
        let test: Vec<_> = pairs.iter().filter(|p| p.role == "test").collect();
        let copy = average(
            &test
                .iter()
                .map(|p| simple_pair_baseline(p, false))
                .collect::<Vec<_>>(),
        );
        let first = average(
            &test
                .iter()
                .map(|p| simple_pair_baseline(p, true))
                .collect::<Vec<_>>(),
        );
        assert!(copy < 0.50, "copy baseline is too strong");
        assert!(first < 0.50, "first-op baseline is too strong");
    }
}
