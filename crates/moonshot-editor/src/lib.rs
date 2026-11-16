use moonshot_core::{
    Candidate, EditPlan, OptimizerType, SamplingPolicy, SchedulePolicy, TaskExample, TaskSplit,
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
    SetAnchorRehearsal { ratio: f32 },
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
            EditProposalAction::SetAnchorRehearsal { ratio } => {
                if *ratio < 0.0 || *ratio > 0.5 {
                    return Err(format!("anchor_rehearsal_ratio {} out of bounds [0.0, 0.5]", ratio));
                }
                updated.anchor_rehearsal_ratio = *ratio;
            }
        }
        Ok(updated)
    }
}

/// Learned Proposal Policy Head mapping state s -> discrete action distribution pi(a | s)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalPolicy {
    pub weights: Vec<f32>, // N_ACTIONS x STATE_DIM
    pub bias: Vec<f32>,    // N_ACTIONS
    pub running_baseline: f32,
    pub step_count: usize,
    pub meta_lr: f32,
}

impl ProposalPolicy {
    pub const N_ACTIONS: usize = 10;
    pub const STATE_DIM: usize = 12;

    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scale = 0.05;
        let weights = (0..Self::N_ACTIONS * Self::STATE_DIM)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let bias = vec![0.0; Self::N_ACTIONS];
        Self {
            weights,
            bias,
            running_baseline: 0.0,
            step_count: 0,
            meta_lr: 0.05,
        }
    }

    pub fn construct_state(
        plan: &EditPlan,
        feedback: &FeedbackSummary,
        val_token_acc: f32,
        loss_slope: f32,
    ) -> Vec<f32> {
        let mut s = vec![0.0; Self::STATE_DIM];
        s[0] = feedback.validation_accuracy.clamp(0.0, 1.0);
        s[1] = val_token_acc.clamp(0.0, 1.0);
        s[2] = feedback.final_loss.tanh();
        s[3] = loss_slope.tanh();
        s[4] = ((plan.base_lr.log10() + 5.0) / 4.0).clamp(0.0, 1.0);
        s[5] = (plan.inner_steps as f32 / 100.0).clamp(0.0, 1.0);
        s[6] = match plan.sampling_policy {
            SamplingPolicy::Uniform => 0.0,
            SamplingPolicy::ErrorWeighted => 0.5,
            SamplingPolicy::HardMining => 1.0,
        };
        s[7] = match plan.schedule_policy {
            SchedulePolicy::Constant => 0.0,
            SchedulePolicy::CosineDecay => 0.5,
            SchedulePolicy::WarmupCosine => 1.0,
        };
        s[8] = match plan.optimizer_type {
            OptimizerType::AdamW => 0.0,
            OptimizerType::Momentum => 0.5,
            _ => 1.0,
        };
        s[9] = if feedback.target_reached { 1.0 } else { 0.0 };
        s[10] = (feedback.samples_used as f32 / 200.0).clamp(0.0, 1.0);
        s[11] = 1.0;
        s
    }

    pub fn action_probabilities(&self, state: &[f32]) -> Vec<f32> {
        let mut logits = self.bias.clone();
        for a in 0..Self::N_ACTIONS {
            let row = &self.weights[a * Self::STATE_DIM..(a + 1) * Self::STATE_DIM];
            for i in 0..Self::STATE_DIM {
                logits[a] += row[i] * state[i];
            }
        }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        logits.iter().map(|&l| (l - max_l).exp() / exp_sum.max(1e-9)).collect()
    }

    pub fn sample_action(&self, state: &[f32], rng: &mut ChaCha8Rng) -> (usize, f32) {
        let probs = self.action_probabilities(state);
        let mut r = rng.gen::<f32>();
        for (a, &p) in probs.iter().enumerate() {
            if r <= p {
                return (a, p);
            }
            r -= p;
        }
        (Self::N_ACTIONS - 1, probs[Self::N_ACTIONS - 1])
    }

    pub fn action_to_proposal(action_idx: usize, current_plan: &EditPlan) -> EditProposalAction {
        match action_idx {
            0 => EditProposalAction::AdjustBaseLr { factor: 1.25 },
            1 => EditProposalAction::AdjustBaseLr { factor: 0.80 },
            2 => EditProposalAction::SwitchSchedule { schedule: SchedulePolicy::CosineDecay },
            3 => EditProposalAction::SwitchSchedule { schedule: SchedulePolicy::WarmupCosine },
            4 => EditProposalAction::SetSamplingPolicy { policy: SamplingPolicy::ErrorWeighted },
            5 => EditProposalAction::SetLayerLrMultipliers {
                multipliers: vec![0.6, 0.8, 1.0, 1.2, 1.4, 1.8],
            },
            6 => EditProposalAction::SetLayerLrMultipliers {
                multipliers: vec![1.8, 1.4, 1.2, 1.0, 0.8, 0.6],
            },
            7 => EditProposalAction::SetInnerSteps { steps: (current_plan.inner_steps + 10).min(90) },
            8 => EditProposalAction::SetInnerSteps { steps: (current_plan.inner_steps.saturating_sub(10)).max(20) },
            _ => EditProposalAction::AdjustBaseLr { factor: 1.0 },
        }
    }

    pub fn update_policy(&mut self, state: &[f32], action_idx: usize, reward: f32) {
        self.step_count += 1;
        let advantage = reward - self.running_baseline;
        self.running_baseline = 0.9 * self.running_baseline + 0.1 * reward;

        let probs = self.action_probabilities(state);
        for a in 0..Self::N_ACTIONS {
            let target_ind = if a == action_idx { 1.0 } else { 0.0 };
            let d_logit = advantage * (target_ind - probs[a]);
            self.bias[a] += self.meta_lr * d_logit;
            let offset = a * Self::STATE_DIM;
            for i in 0..Self::STATE_DIM {
                self.weights[offset + i] += self.meta_lr * d_logit * state[i];
            }
        }
    }
}

/// Wave 2: Learned Credit Routing & Plasticity Gating Head (Leap B1)
/// Maps state s in R^12 -> per-group plasticity gates gamma in [0, 1]^6
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityRouter {
    pub weights: Vec<f32>, // 6 x 12 = 72
    pub bias: Vec<f32>,    // 6
    pub meta_lr: f32,
    pub step_count: usize,
}

impl PlasticityRouter {
    pub const N_GROUPS: usize = 6;
    pub const STATE_DIM: usize = 12;

    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scale = 0.05;
        let weights = (0..Self::N_GROUPS * Self::STATE_DIM)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let bias = vec![0.5; Self::N_GROUPS]; // initialize slightly positive (sigmoid ~0.62)
        Self {
            weights,
            bias,
            meta_lr: 0.05,
            step_count: 0,
        }
    }

    pub fn compute_gates(&self, state: &[f32]) -> Vec<f32> {
        let mut gates = Vec::with_capacity(Self::N_GROUPS);
        for g in 0..Self::N_GROUPS {
            let mut z = self.bias[g];
            let row = &self.weights[g * Self::STATE_DIM..(g + 1) * Self::STATE_DIM];
            for i in 0..Self::STATE_DIM {
                z += row[i] * state[i];
            }
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            gates.push(sigmoid);
        }
        gates
    }

    pub fn update(&mut self, state: &[f32], gates: &[f32], advantage: f32) {
        self.step_count += 1;
        for g in 0..Self::N_GROUPS {
            let gamma = gates[g];
            let d_z = advantage * gamma * (1.0 - gamma);
            self.bias[g] += self.meta_lr * d_z;
            let offset = g * Self::STATE_DIM;
            for i in 0..Self::STATE_DIM {
                self.weights[offset + i] += self.meta_lr * d_z * state[i];
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditorMode {
    Learned(ProposalPolicy),
    Frozen(ProposalPolicy),
    ShuffledFeedback(ProposalPolicy),
    Heuristic,
    Random,
}

pub struct EditorEngine;

impl EditorEngine {
    pub fn propose_edit(
        mode: &mut EditorMode,
        current_plan: &EditPlan,
        feedback: &FeedbackSummary,
        val_token_acc: f32,
        loss_slope: f32,
        seed: u64,
    ) -> Result<(EditPlan, Option<(Vec<f32>, usize)>), String> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        match mode {
            EditorMode::Learned(policy) => {
                let state = ProposalPolicy::construct_state(current_plan, feedback, val_token_acc, loss_slope);
                let (action_idx, _) = policy.sample_action(&state, &mut rng);
                let proposal = ProposalPolicy::action_to_proposal(action_idx, current_plan);
                let updated = SandboxTypechecker::validate_and_apply(current_plan, &proposal)?;
                Ok((updated, Some((state, action_idx))))
            }
            EditorMode::Frozen(policy) => {
                let state = ProposalPolicy::construct_state(current_plan, feedback, val_token_acc, loss_slope);
                let (action_idx, _) = policy.sample_action(&state, &mut rng);
                let proposal = ProposalPolicy::action_to_proposal(action_idx, current_plan);
                let updated = SandboxTypechecker::validate_and_apply(current_plan, &proposal)?;
                Ok((updated, None))
            }
            EditorMode::ShuffledFeedback(policy) => {
                let mut state = ProposalPolicy::construct_state(current_plan, feedback, val_token_acc, loss_slope);
                // Permute state indices to destroy causal observation feedback
                for i in 0..state.len() {
                    let j = rng.gen_range(0..state.len());
                    state.swap(i, j);
                }
                let (action_idx, _) = policy.sample_action(&state, &mut rng);
                let proposal = ProposalPolicy::action_to_proposal(action_idx, current_plan);
                let updated = SandboxTypechecker::validate_and_apply(current_plan, &proposal)?;
                Ok((updated, Some((state, action_idx))))
            }
            EditorMode::Random => {
                let action_idx = rng.gen_range(0..ProposalPolicy::N_ACTIONS);
                let proposal = ProposalPolicy::action_to_proposal(action_idx, current_plan);
                let updated = SandboxTypechecker::validate_and_apply(current_plan, &proposal)?;
                Ok((updated, None))
            }
            EditorMode::Heuristic => {
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
                let chosen = &candidate_actions[rng.gen_range(0..candidate_actions.len())];
                let updated = SandboxTypechecker::validate_and_apply(current_plan, chosen)?;
                Ok((updated, None))
            }
        }
    }
}

pub struct SearchManager {
    pub incumbent: Candidate,
    pub base_model_checkpoint: ModelCheckpoint,
    pub editor_mode: EditorMode,
    pub candidate_history: Vec<TrialRecord>,
    pub promotion_count: usize,
    pub rollback_count: usize,
    pub family_baselines: std::collections::HashMap<String, f32>,
}

impl SearchManager {
    pub fn new(initial_model: &TinyTransformer, initial_plan: EditPlan, editor_mode: EditorMode) -> Self {
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
            editor_mode,
            candidate_history: Vec::new(),
            promotion_count: 0,
            rollback_count: 0,
            family_baselines: std::collections::HashMap::new(),
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

        let policy_path = dir.join("editor_policy.json");
        let policy_json = serde_json::to_string_pretty(&self.editor_mode).map_err(|e| e.to_string())?;
        fs::write(&policy_path, policy_json).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn load_state_from_dir(dir: &Path, model: &mut TinyTransformer) -> Result<Self, String> {
        let ckpt_path = dir.join("best_checkpoint.json");
        let model_path = dir.join("best_model_checkpoint.json");
        let policy_path = dir.join("editor_policy.json");

        if !ckpt_path.exists() || !model_path.exists() {
            return Err("Checkpoint files missing in specified directory".to_string());
        }

        let ckpt_str = fs::read_to_string(&ckpt_path).map_err(|e| e.to_string())?;
        let incumbent: Candidate = serde_json::from_str(&ckpt_str).map_err(|e| e.to_string())?;

        let model_str = fs::read_to_string(&model_path).map_err(|e| e.to_string())?;
        let base_model_checkpoint: ModelCheckpoint = serde_json::from_str(&model_str).map_err(|e| e.to_string())?;

        model.load_checkpoint(&base_model_checkpoint)?;

        let editor_mode = if policy_path.exists() {
            let p_str = fs::read_to_string(&policy_path).map_err(|e| e.to_string())?;
            serde_json::from_str(&p_str).unwrap_or_else(|_| EditorMode::Learned(ProposalPolicy::new(42)))
        } else {
            EditorMode::Learned(ProposalPolicy::new(42))
        };

        Ok(Self {
            incumbent,
            base_model_checkpoint,
            editor_mode,
            candidate_history: Vec::new(),
            promotion_count: 0,
            rollback_count: 0,
            family_baselines: std::collections::HashMap::new(),
        })
    }

    pub fn evaluate_adaptation_with_anchors(
        model: &mut TinyTransformer,
        plan: &EditPlan,
        split: &TaskSplit,
        target_acc: f32,
        anchor_examples: Option<&[TaskExample]>,
    ) -> (Option<usize>, f32, f32, f32) {
        let mut samples_to_target = None;
        let batch_size = 8;
        let mut total_samples_seen = 0;
        let total_steps = plan.inner_steps;
        let mut final_loss = 0.0;

        for step in 0..total_steps {
            // Interleaved anchor rehearsal step to maintain retention
            if plan.anchor_rehearsal_ratio > 0.0 && step > 0 && (step % 5 == 0) {
                if let Some(anchors) = anchor_examples {
                    if !anchors.is_empty() {
                        let a_start = (step * batch_size) % anchors.len();
                        let a_end = (a_start + batch_size).min(anchors.len());
                        let a_batch = &anchors[a_start..a_end];
                        model.train_step(a_batch, plan, step, total_steps);
                    }
                }
            }

            let start_idx = (step * batch_size) % split.train_examples.len().max(1);
            let end_idx = (start_idx + batch_size).min(split.train_examples.len());
            let batch = &split.train_examples[start_idx..end_idx];

            final_loss = model.train_step_projected(batch, anchor_examples, plan, step, total_steps);
            total_samples_seen += batch.len();

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

        let mut test_preds = Vec::with_capacity(split.test_examples.len());
        for test_ex in &split.test_examples {
            let pred = model.predict_sequence(&test_ex.input_tokens, test_ex.target_tokens.len());
            test_preds.push(pred);
        }
        let final_exact = ImmutableOracle::score_batch_exact(&test_preds, &split.test_examples);
        let final_token = ImmutableOracle::score_batch_tokens(&test_preds, &split.test_examples);

        (samples_to_target, final_exact, final_token, final_loss)
    }

    pub fn evaluate_adaptation(
        model: &mut TinyTransformer,
        plan: &EditPlan,
        split: &TaskSplit,
        target_acc: f32,
    ) -> (Option<usize>, f32, f32, f32) {
        Self::evaluate_adaptation_with_anchors(model, plan, split, target_acc, None)
    }

    pub fn run_trial(
        &mut self,
        model: &mut TinyTransformer,
        proposed_plan: EditPlan,
        policy_trace: Option<(Vec<f32>, usize)>,
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

        // Per-family baseline tracking to prevent easier tasks from blocking harder tasks
        let pre_val_exact = self.family_baselines.get(&split.family_name).cloned().unwrap_or(0.0);

        let start_time = std::time::Instant::now();
        let (samples_to_target, final_exact, final_token, _) = Self::evaluate_adaptation(model, &proposed_plan, split, target_acc);
        let elapsed_secs = start_time.elapsed().as_secs_f64();

        let target_reached = samples_to_target.is_some();
        let baseline_samples = 200;

        // Compute reward for policy gradient update
        let delta_exact = final_exact - pre_val_exact;
        let step_cost = 0.02 * (proposed_plan.inner_steps as f32 / 100.0);
        let reward = delta_exact + 0.2 * final_token - step_cost;

        // Update proposal policy if in learned mode
        if let Some((state, action_idx)) = policy_trace {
            match &mut self.editor_mode {
                EditorMode::Learned(policy) | EditorMode::ShuffledFeedback(policy) => {
                    policy.update_policy(&state, action_idx, reward);
                }
                _ => {}
            }
        }

        let is_non_regressive = final_exact >= (pre_val_exact - 0.001);
        let is_faster = target_reached && samples_to_target.unwrap_or(usize::MAX) <= baseline_samples;
        let is_significantly_better = final_exact > (pre_val_exact + 0.02);

        let should_promote = is_non_regressive && (is_faster || is_significantly_better);

        let status = if should_promote {
            TrialStatus::Promoted
        } else {
            TrialStatus::Rejected
        };

        if should_promote {
            self.promotion_count += 1;
            self.family_baselines.insert(split.family_name.clone(), final_exact);
            self.incumbent = Candidate {
                id: candidate_id.clone(),
                weights_checksum: model.checksum(),
                learning_rule: proposed_plan.clone(),
                meta_generation: self.incumbent.meta_generation + 1,
                validated_accuracy: final_exact,
                adaptation_history: {
                    let mut h = self.incumbent.adaptation_history.clone();
                    h.push(format!("promoted in {} (parent: {}, exact: {:.3}, token: {:.3}, reward: {:.3})", trial_id, parent_id, final_exact, final_token, reward));
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

    #[test]
    fn test_learned_proposal_policy_update() {
        let mut policy = ProposalPolicy::new(42);
        let feedback = FeedbackSummary {
            final_loss: 0.5,
            validation_accuracy: 0.6,
            samples_used: 50,
            target_reached: true,
            error_count: 2,
        };
        let plan = EditPlan::default();
        let state = ProposalPolicy::construct_state(&plan, &feedback, 0.8, -0.2);
        let probs_before = policy.action_probabilities(&state);

        // Positive reinforcement on action 0
        policy.update_policy(&state, 0, 1.5);
        let probs_after = policy.action_probabilities(&state);

        assert!(
            probs_after[0] > probs_before[0],
            "Reinforced action probability must increase: before={}, after={}",
            probs_before[0], probs_after[0]
        );
    }
}
