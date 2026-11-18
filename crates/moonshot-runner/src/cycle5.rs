use crate::cycle5_model::Model;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Int, Tensor, TensorData};
use moonshot_core::TaskExample;
use moonshot_model::{ModelConfig, TinyTransformer};
use moonshot_oracle::{BOS_TOKEN, CMD_TOOL_COMMIT, CMD_TOOL_SET, EOS_TOKEN, SEP_TOKEN, VAL_BASE};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

const PARAMETERS: usize = 1_095_744;
const PARAMETER_TENSORS: usize = 38;
const LEARNING_RATE: f64 = 0.001;
const TRAIN_STEPS: usize = 300;
const SMOKE_STEPS: usize = 8;
const INPUT_LEN: usize = 5;
const TARGET_LEN: usize = 5;
const TRAIN_LEN: usize = INPUT_LEN + TARGET_LEN - 1;
const PARITY_TOLERANCE: f32 = 0.005;

fn example(register: usize, value: usize) -> TaskExample {
    let set = [CMD_TOOL_SET, VAL_BASE + register, VAL_BASE + value];
    TaskExample {
        input_tokens: [vec![BOS_TOKEN], set.to_vec(), vec![SEP_TOKEN]].concat(),
        target_tokens: [set.to_vec(), vec![CMD_TOOL_COMMIT, EOS_TOKEN]].concat(),
    }
}

fn dataset(seed: u64) -> Vec<TaskExample> {
    let mut examples: Vec<_> = (0..4)
        .flat_map(|register| (1..=9).map(move |value| example(register, value)))
        .collect();
    examples.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    examples
}

fn tokens(
    values: impl IntoIterator<Item = usize>,
    batch: usize,
    len: usize,
    device: &Device,
) -> Tensor<2, Int> {
    let data: Vec<_> = values.into_iter().map(|token| token as i64).collect();
    Tensor::from_data(TensorData::new(data, [batch, len]), device)
}

fn evaluate(
    model: &Model,
    examples: &[TaskExample],
    config: &ModelConfig,
    device: &Device,
) -> (f64, Vec<Vec<usize>>) {
    let batch = examples.len();
    let mut sequences: Vec<_> = examples
        .iter()
        .map(|example| example.input_tokens.clone())
        .collect();
    for len in INPUT_LEN..INPUT_LEN + TARGET_LEN {
        let input = tokens(sequences.iter().flatten().copied(), batch, len, device);
        let predicted = model
            .forward(input, config.n_heads)
            .slice([0..batch, len - 1..len, 0..config.vocab_size])
            .argmax(2)
            .to_data_as::<i64>()
            .try_into_vec::<i64>()
            .unwrap();
        for (sequence, token) in sequences.iter_mut().zip(predicted) {
            sequence.push(token as usize);
        }
    }
    let predictions: Vec<_> = sequences
        .into_iter()
        .map(|sequence| sequence[INPUT_LEN..].to_vec())
        .collect();
    let correct = predictions
        .iter()
        .zip(examples)
        .filter(|(prediction, example)| **prediction == example.target_tokens)
        .count();
    (correct as f64 / batch as f64, predictions)
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn run(
    seed: u64,
    output: &Path,
    commit: &str,
    target: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("expected a full lowercase Git SHA".into());
    }
    match target {
        Some(0.85) if [4109, 4201, 4303].contains(&seed) => (),
        None if seed == 42 => (),
        _ => return Err("seed or target does not match the preregistered protocol".into()),
    }

    fs::create_dir_all(output)?;
    let device = Device::cuda(0);
    let train_device = device.clone().autodiff();
    let config = ModelConfig::new_1m();
    let mut native = TinyTransformer::new(config.clone(), seed);
    let mut model = Model::from_moonshot(&native, &train_device);
    assert_eq!(model.num_params(), PARAMETERS);
    let initial_weights = model.parameters();
    assert_eq!(initial_weights, native.extract_parameters());
    let examples = if target.is_some() {
        dataset(seed)
    } else {
        vec![example(0, 1), example(3, 9)]
    };
    let (train, held_out) = examples.split_at(if target.is_some() { 24 } else { examples.len() });
    let batch = train.len();
    let input = tokens(
        train.iter().flat_map(|example| {
            example
                .input_tokens
                .iter()
                .chain(&example.target_tokens[..TARGET_LEN - 1])
                .copied()
        }),
        batch,
        TRAIN_LEN,
        &train_device,
    );
    let labels = tokens(
        train
            .iter()
            .flat_map(|example| example.target_tokens.iter().copied()),
        batch,
        TARGET_LEN,
        &train_device,
    )
    .reshape([batch * TARGET_LEN]);

    let parity_error = if target.is_none() {
        let expected: Vec<_> = train
            .iter()
            .flat_map(|example| {
                let full = [
                    &example.input_tokens[..],
                    &example.target_tokens[..TARGET_LEN - 1],
                ]
                .concat();
                native.forward(&full)
            })
            .collect();
        let actual = model
            .forward(input.clone(), config.n_heads)
            .to_data_as::<f32>()
            .try_into_vec::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f32;
        for (cuda, cpu) in actual.into_iter().zip(expected) {
            assert!(cuda.is_finite() && cpu.is_finite());
            max_error = max_error.max((cuda - cpu).abs());
        }
        assert!(
            max_error <= PARITY_TOLERANCE,
            "CPU/CUDA logit error {max_error}"
        );
        Some(max_error)
    } else {
        None
    };

    let objective = CrossEntropyLossConfig::new().init(&train_device);
    let loss = |model: &Model| {
        let logits = model
            .forward(input.clone(), config.n_heads)
            .slice([0..batch, INPUT_LEN - 1..TRAIN_LEN, 0..config.vocab_size])
            .reshape([batch * TARGET_LEN, config.vocab_size]);
        objective.forward(logits, labels.clone())
    };
    let mut optimizer = AdamConfig::new().init();
    let steps = if target.is_some() {
        TRAIN_STEPS
    } else {
        SMOKE_STEPS
    };
    let mut losses = Vec::with_capacity(steps + 1);
    for step in 0..steps {
        let step_loss = loss(&model);
        let value: f32 = step_loss.clone().into_scalar();
        assert!(value.is_finite(), "non-finite loss at step {step}");
        losses.push(value);
        let gradients = GradientsParams::from_grads(step_loss.backward(), &model);
        assert_eq!(
            gradients.len(),
            PARAMETER_TENSORS,
            "missing parameter gradients"
        );
        model = optimizer.step(LEARNING_RATE, model, gradients);
        if step % 10 == 0 || step + 1 == steps {
            println!(
                "CUDA seed={seed} update={}/{} loss={value:.6}",
                step + 1,
                steps
            );
        }
    }
    let final_loss: f32 = loss(&model).into_scalar();
    assert!(final_loss.is_finite());
    let initial_loss = losses[0];
    if target.is_none() {
        assert!(
            final_loss < initial_loss,
            "controlled batch loss did not decrease"
        );
    }
    losses.push(final_loss);
    let weights = model.parameters();
    assert_eq!(weights.len(), PARAMETERS);
    assert!(weights.iter().all(|weight| weight.is_finite()));
    assert_ne!(initial_weights, weights, "optimizer did not change weights");

    let mut result = json!({
        "backend": "burn-cuda", "model_parameters": PARAMETERS, "git_commit": commit,
        "gpu_forward": true, "gpu_backward": true, "gpu_optimizer_update": true,
        "optimizer_updates": steps, "weights_changed": true,
        "gradient_tensors_per_step": PARAMETER_TENSORS,
        "initial_loss": initial_loss, "final_loss": final_loss, "loss_by_step": losses,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": batch,
        "training_steps": steps, "seed": seed
    });
    let filename = if let Some(target) = target {
        let (validation, test) = held_out.split_at(6);
        native.load_parameters(&weights);
        let checkpoint = native.to_safetensors()?;
        let checkpoint_file = "model.safetensors";
        fs::write(output.join(checkpoint_file), &checkpoint)?;
        let inference_model = Model::from_moonshot(&native, &device);
        let splits = json!({"train": train, "validation": validation, "test": test});
        let split_bytes = serde_json::to_vec(&splits)?;
        fs::write(output.join("dataset_splits.json"), &split_bytes)?;
        result["dataset_split_sha256"] = hash(&split_bytes).into();
        result["dataset_sizes"] =
            json!({"train": train.len(), "validation": validation.len(), "test": test.len()});
        result["split_sha256"] = json!({
            "train": hash(&serde_json::to_vec(&train)?),
            "validation": hash(&serde_json::to_vec(&validation)?),
            "test": hash(&serde_json::to_vec(&test)?)
        });
        let mut predictions = serde_json::Map::new();
        for (split, examples) in [("train", train), ("validation", validation), ("test", test)] {
            let (score, output_tokens) = evaluate(&inference_model, examples, &config, &device);
            result[format!("{split}_exact_match")] = score.into();
            predictions.insert(split.into(), json!(output_tokens));
        }
        fs::write(
            output.join("predictions.json"),
            serde_json::to_vec(&predictions)?,
        )?;
        result["task_family"] = "tool-workflow-basic".into();
        result["target_acc"] = target.into();
        result["checkpoint_file"] = checkpoint_file.into();
        result["checkpoint_sha256"] = hash(&checkpoint).into();
        "capability_result.json"
    } else {
        result["cpu_gpu_parity_passed"] = true.into();
        result["cpu_gpu_max_logit_error"] = json!(parity_error);
        "smoke_result.json"
    };
    fs::write(output.join(filename), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registered_splits_are_reproducible_and_disjoint() {
        for seed in [4109, 4201, 4303] {
            let examples = dataset(seed);
            let unique: HashSet<_> = examples
                .iter()
                .map(|example| &example.input_tokens)
                .collect();
            assert_eq!(unique.len(), 36);
            assert_eq!(
                serde_json::to_vec(&examples).unwrap(),
                serde_json::to_vec(&dataset(seed)).unwrap()
            );
        }
    }
}
