use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_count: usize,
    pub ram_bytes: u64,
    pub gpu_name: Option<String>,
    pub gpu_memory_bytes: Option<u64>,
    pub cuda_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLedger {
    pub run_id: String,
    pub git_commit: String,
    pub hardware: HardwareInfo,
    pub rust_version: String,
    pub backend: String,
    pub start_time: DateTime<Utc>,
    pub duration_seconds: f64,
    pub gpu_minutes: f64,
    pub artifacts: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialStatus {
    Promoted,
    Rejected,
    Crashed,
    BudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    Sgd,
    Momentum,
    AdamW,
    SignGradient,
    SparseMasked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingPolicy {
    Uniform,
    ErrorWeighted,
    HardMining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulePolicy {
    Constant,
    CosineDecay,
    WarmupCosine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EditPlan {
    pub optimizer_type: OptimizerType,
    pub base_lr: f32,
    pub layer_lr_multipliers: Vec<f32>,
    pub sampling_policy: SamplingPolicy,
    pub schedule_policy: SchedulePolicy,
    pub weight_decay: f32,
    pub grad_clip: Option<f32>,
    pub inner_steps: usize,
    pub sparse_gradient_threshold: Option<f32>,
    #[serde(default)]
    pub anchor_rehearsal_ratio: f32,
}

impl Default for EditPlan {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            base_lr: 0.015,
            layer_lr_multipliers: vec![1.0; 6],
            sampling_policy: SamplingPolicy::Uniform,
            schedule_policy: SchedulePolicy::CosineDecay,
            weight_decay: 0.001,
            grad_clip: Some(1.0),
            inner_steps: 35,
            sparse_gradient_threshold: None,
            anchor_rehearsal_ratio: 0.0,
        }
    }
}

impl EditPlan {
    pub fn checksum(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExample {
    pub input_tokens: Vec<usize>,
    pub target_tokens: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSplit {
    pub family_name: String,
    pub is_held_out: bool,
    pub train_examples: Vec<TaskExample>,
    pub val_examples: Vec<TaskExample>,
    pub test_examples: Vec<TaskExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecord {
    pub trial_id: String,
    pub hypothesis_id: String,
    pub candidate_id: String,
    pub parent_candidate_id: Option<String>,
    pub task_family: String,
    pub seed: u64,
    pub edit_plan: EditPlan,
    pub samples_to_target: Option<usize>,
    pub target_accuracy_reached: bool,
    pub target_accuracy: f32,
    pub final_accuracy: f32,
    pub baseline_samples_to_target: Option<usize>,
    pub relative_sample_efficiency: Option<f32>,
    pub inner_steps_used: usize,
    pub gpu_seconds: f64,
    pub status: TrialStatus,
    pub failure_reason: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub weights_checksum: String,
    pub learning_rule: EditPlan,
    pub meta_generation: usize,
    pub validated_accuracy: f32,
    pub adaptation_history: Vec<String>,
}

pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}
