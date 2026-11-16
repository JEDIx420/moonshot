use moonshot_core::{
    Candidate, EditPlan, OptimizerType, SamplingPolicy, SchedulePolicy, TaskSplit,
    TrialRecord, TrialStatus,
};
use moonshot_model::{ModelCheckpoint, TinyTransformer};
use moonshot_oracle::ImmutableOracle;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSummary {
    pub final_loss: f32,
    pub validation_accuracy: f32,
    pub samples_used: usize,
    pub target_reached: bool,
    pub error_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditProposalAction {
    AdjustBaseLr { factor: f32 },
    SetLayerLrMultipliers { multipliers: Vec<f32> },
    SwitchOptimizer { optimizer: OptimizerType },
    SwitchSchedule { schedule: SchedulePolicy },
    SetSamplingPolicy { policy: SamplingPolicy },
    SetSparseGradientThreshold { threshold: Option<f32> },
    AdjustWeightDecay { decay: f32 },
    SetInnerSteps { steps: usize },
}

pub struct SandboxTypechecker;

impl SandboxTypechecker {
    pub const MIN_LR: f32 = 1e-5;
    pub const MAX_LR: f32 = 0.1;
    pub const MIN_INNER_STEPS: usize = 1;
    pub const MAX_INNER_STEPS: usize = 100;

    pub fn validate_and_apply(current: &EditPlan, action: &EditProposalAction) -> Result<EditPlan, String> {
        let mut updated = current.clone();
        match action {
            EditProposalAction::AdjustBaseLr { factor } => {
                let new_lr = current.base_lr * factor;
                if new_lr < Self::MIN_LR || new_lr > Self::MAX_LR {
                    return Err(format!("base_lr {} out of safe bounds [{}, {}]", new_lr, Self::MIN_LR, Self::MAX_LR));
                }
                updated.base_lr = new_lr;
            }
            EditProposalAction::SetLayerLrMultipliers { multipliers } => {
                if multipliers.len() != current.layer_lr_multipliers.len() {
                    return Err(format!("layer_lr_multipliers length mismatch: expected {}, got {}", current.layer_lr_multipliers.len(), multipliers.len()));
                }
                for &m in multipliers {
                    if m < 0.0 || m > 5.0 {
                        return Err(format!("multiplier {} out of safe bounds [0.0, 5.0]", m));
                    }
                }
                updated.layer_lr_multipliers = multipliers.clone();
            }
            EditProposalAction::SwitchOptimizer { optimizer } => {
                updated.optimizer_type = *optimizer;
            }
            EditProposalAction::SwitchSchedule { schedule } => {
                updated.schedule_policy = *schedule;
            }
            EditProposalAction::SetSamplingPolicy { policy } => {
                updated.sampling_policy = *policy;
            }
            EditProposalAction::SetSparseGradientThreshold { threshold } => {
                if let Some(th) = threshold {
                    if *th <= 0.0 || *th > 1.0 {
                        return Err(format!("sparse threshold {} out of safe bounds (0.0, 1.0]", th));
                    }
                }
                updated.sparse_gradient_threshold = *threshold;
            }
            EditProposalAction::AdjustWeightDecay { decay } => {
                if *decay < 0.0 || *decay > 0.5 {
                    return Err(format!("weight_decay {} out of safe bounds [0.0, 0.5]", decay));
                }
                updated.weight_decay = *decay;
            }
            EditProposalAction::SetInnerSteps { steps } => {
                if *steps < Self::MIN_INNER_STEPS || *steps > Self::MAX_INNER_STEPS {
                    return Err(format!("inner_steps {} out of bounds [{}, {}]", steps, Self::MIN_INNER_STEPS, Self::MAX_INNER_STEPS));
                }
                updated.inner_steps = *steps;
            }
        }
        Ok(updated)
    }
}

pub struct EditorEngine;

impl EditorEngine {
    // Propose an EditPlan mutation conditioned on past trial feedback
    pub fn propose_edit(current_plan: &EditPlan, feedback: &FeedbackSummary, seed: u64) -> Result<EditPlan, String> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let candidate_actions = if !feedback.target_reached && feedback.final_loss > 1.5 {
            vec![
                EditProposalAction::AdjustBaseLr { factor: 1.25 },
                EditProposalAction::SwitchOptimizer { optimizer: OptimizerType::AdamW },
                EditProposalAction::SwitchSchedule { schedule: SchedulePolicy::WarmupCosine },
                EditProposalAction::SetSamplingPolicy { policy: SamplingPolicy::ErrorWeighted },
                EditProposalAction::SetInnerSteps { steps: (current_plan.inner_steps + 10).min(90) },
            ]
        } else if feedback.target_reached {
            vec![
                EditProposalAction::SetSparseGradientThreshold { threshold: Some(0.01) },
                EditProposalAction::SetLayerLrMultipliers {
                    multipliers: vec![0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
                },
                EditProposalAction::SetSamplingPolicy { policy: SamplingPolicy::ErrorWeighted },
                EditProposalAction::AdjustWeightDecay { decay: 0.005 },
            ]
        } else {
            vec![
                EditProposalAction::SetLayerLrMultipliers {
                    multipliers: vec![0.6, 0.8, 1.0, 1.2, 1.4, 1.8],
                },
                EditProposalAction::AdjustBaseLr { factor: 1.1 },
                EditProposalAction::SwitchSchedule { schedule: SchedulePolicy::CosineDecay },
                EditProposalAction::SetInnerSteps { steps: (current_plan.inner_steps + 5).min(80) },
            ]
        };

        let chosen_action = &candidate_actions[rng.gen_range(0..candidate_actions.len())];
        SandboxTypechecker::validate_and_apply(current_plan, chosen_action)
    }
}

pub struct SearchManager {
    pub incumbent: Candidate,
    pub base_model_checkpoint: ModelCheckpoint,
    pub candidate_history: Vec<TrialRecord>,
    pub promotion_count: usize,
    pub rollback_count: usize,
}

impl SearchManager {
    pub fn new(initial_model: &TinyTransformer, initial_plan: EditPlan) -> Self {
        let base_ckpt = initial_model.save_checkpoint();
        let incumbent = Candidate {
            id: "cand_init".to_string(),
            weights_checksum: initial_model.checksum(),
            learning_rule: initial_plan,
            meta_generation: 0,
            validated_accuracy: 0.0,
            adaptation_history: vec!["initial_baseline".to_string()],
        };

        Self {
            incumbent,
            base_model_checkpoint: base_ckpt,
            candidate_history: Vec::new(),
            promotion_count: 0,
            rollback_count: 0,
        }
    }

    pub fn save_state_to_dir(&self, dir: &Path) -> Result<(), String> {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;

        let ckpt_path = dir.join("best_checkpoint.json");
        let ckpt_json = serde_json::to_string_pretty(&self.incumbent).map_err(|e| e.to_string())?;
        fs::write(&ckpt_path, ckpt_json).map_err(|e| e.to_string())?;

        let model_path = dir.join("best_model_checkpoint.json");
        let model_json = serde_json::to_string_pretty(&self.base_model_checkpoint).map_err(|e| e.to_string())?;
        fs::write(&model_path, model_json).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn load_state_from_dir(dir: &Path, model: &mut TinyTransformer) -> Result<Self, String> {
        let ckpt_path = dir.join("best_checkpoint.json");
        let model_path = dir.join("best_model_checkpoint.json");

        if !ckpt_path.exists() || !model_path.exists() {
            return Err("Checkpoint files missing in specified directory".to_string());
        }

        let ckpt_str = fs::read_to_string(&ckpt_path).map_err(|e| e.to_string())?;
        let incumbent: Candidate = serde_json::from_str(&ckpt_str).map_err(|e| e.to_string())?;

        let model_str = fs::read_to_string(&model_path).map_err(|e| e.to_string())?;
        let base_model_checkpoint: ModelCheckpoint = serde_json::from_str(&model_str).map_err(|e| e.to_string())?;

        model.load_checkpoint(&base_model_checkpoint)?;

        Ok(Self {
            incumbent,
            base_model_checkpoint,
            candidate_history: Vec::new(),
            promotion_count: 0,
            rollback_count: 0,
        })
    }

    // Evaluate adaptation efficiency of candidate on a task split using exact sequence match
    pub fn evaluate_adaptation(
        model: &mut TinyTransformer,
        plan: &EditPlan,
        split: &TaskSplit,
        target_acc: f32,
    ) -> (Option<usize>, f32, f32, f32) {
        let mut samples_to_target = None;
        let batch_size = 8;
        let mut total_samples_seen = 0;
        let total_steps = plan.inner_steps;
        let mut final_loss = 0.0;

        for step in 0..total_steps {
            let start_idx = (step * batch_size) % split.train_examples.len().max(1);
            let end_idx = (start_idx + batch_size).min(split.train_examples.len());
            let batch = &split.train_examples[start_idx..end_idx];

            final_loss = model.train_step(batch, plan, step, total_steps);
            total_samples_seen += batch.len();

            // Evaluate on validation slice using exact sequence match
            let mut preds = Vec::with_capacity(split.val_examples.len());
            for val_ex in &split.val_examples {
                let pred = model.predict_sequence(&val_ex.input_tokens, val_ex.target_tokens.len());
                preds.push(pred);
            }
            let acc = ImmutableOracle::score_batch_exact(&preds, &split.val_examples);
            if acc >= target_acc && samples_to_target.is_none() {
                samples_to_target = Some(total_samples_seen);
            }
        }

        // Final test score evaluated via exact full sequence match and token overlap
        let mut test_preds = Vec::with_capacity(split.test_examples.len());
        for test_ex in &split.test_examples {
            let pred = model.predict_sequence(&test_ex.input_tokens, test_ex.target_tokens.len());
            test_preds.push(pred);
        }
        let final_exact = ImmutableOracle::score_batch_exact(&test_preds, &split.test_examples);
        let final_token = ImmutableOracle::score_batch_tokens(&test_preds, &split.test_examples);

        (samples_to_target, final_exact, final_token, final_loss)
    }

    // Attempt trial, decide promotion or rollback, and record immutable trial record
    pub fn run_trial(
        &mut self,
        model: &mut TinyTransformer,
        proposed_plan: EditPlan,
        split: &TaskSplit,
        seed: u64,
        hypothesis_id: &str,
        target_acc: f32,
    ) -> TrialRecord {
        let trial_idx = self.candidate_history.len() + 1;
        let trial_id = format!("trial_{:04}", trial_idx);
        let parent_id = self.incumbent.id.clone();
        let candidate_id = format!("cand_gen{:02}_{:04}", self.incumbent.meta_generation + 1, trial_idx);

        // Reset task model to base capability initialization before trial
        model.load_checkpoint(&self.base_model_checkpoint).expect("Failed to restore base model checkpoint");

        let start_time = std::time::Instant::now();
        let (samples_to_target, final_exact, final_token, _) = Self::evaluate_adaptation(model, &proposed_plan, split, target_acc);
        let elapsed_secs = start_time.elapsed().as_secs_f64();

        let target_reached = samples_to_target.is_some();
        let baseline_samples = 200; // reference budget

        // Strict non-regression & Pareto promotion criteria
        let is_non_regressive = final_exact >= (self.incumbent.validated_accuracy - 0.001);
        let is_faster = target_reached && samples_to_target.unwrap_or(usize::MAX) <= baseline_samples;
        let is_significantly_better = final_exact > (self.incumbent.validated_accuracy + 0.02);

        let should_promote = is_non_regressive && (is_faster || is_significantly_better);

        let status = if should_promote {
            TrialStatus::Promoted
        } else {
            TrialStatus::Rejected
        };

        if should_promote {
            self.promotion_count += 1;
            self.incumbent = Candidate {
                id: candidate_id.clone(),
                weights_checksum: model.checksum(),
                learning_rule: proposed_plan.clone(),
                meta_generation: self.incumbent.meta_generation + 1,
                validated_accuracy: final_exact,
                adaptation_history: {
                    let mut h = self.incumbent.adaptation_history.clone();
                    h.push(format!("promoted in {} (parent: {}, exact: {:.3}, token: {:.3})", trial_id, parent_id, final_exact, final_token));
                    h
                },
            };
        } else {
            self.rollback_count += 1;
        }

        let record = TrialRecord {
            trial_id,
            hypothesis_id: hypothesis_id.to_string(),
            candidate_id,
            parent_candidate_id: Some(parent_id),
            task_family: split.family_name.clone(),
            seed,
            edit_plan: proposed_plan.clone(),
            samples_to_target,
            target_accuracy_reached: target_reached,
            target_accuracy: target_acc,
            final_accuracy: final_exact,
            baseline_samples_to_target: Some(baseline_samples),
            relative_sample_efficiency: samples_to_target.map(|s| baseline_samples as f32 / s as f32),
            inner_steps_used: proposed_plan.inner_steps,
            gpu_seconds: elapsed_secs,
            status,
            failure_reason: if should_promote {
                None
            } else {
                Some("failed_non_regression_or_efficiency_gate".to_string())
            },
            timestamp: chrono::Utc::now(),
        };

        self.candidate_history.push(record.clone());
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_typechecker_bounds() {
        let plan = EditPlan::default();
        let valid_action = EditProposalAction::AdjustBaseLr { factor: 1.2 };
        assert!(SandboxTypechecker::validate_and_apply(&plan, &valid_action).is_ok());

        let invalid_action = EditProposalAction::AdjustBaseLr { factor: 1000.0 };
        assert!(SandboxTypechecker::validate_and_apply(&plan, &invalid_action).is_err());
    }
}
