use moonshot_core::{TaskExample, TaskSplit};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const VOCAB_SIZE: usize = 64;
pub const PAD_TOKEN: usize = 0;
pub const BOS_TOKEN: usize = 1;
pub const EOS_TOKEN: usize = 2;
pub const SEP_TOKEN: usize = 3;

// Symbol tokens: 4..12 (A=4, B=5, C=6, D=7, E=8, F=9, G=10, H=11)
pub const SYMBOL_BASE: usize = 4;
pub const NUM_SYMBOLS: usize = 8;

// State tokens: 12..20
pub const STATE_BASE: usize = 12;

// Command tokens: 20..32
pub const CMD_REVERSE: usize = 20;
pub const CMD_SHIFT_1: usize = 21;
pub const CMD_SHIFT_2: usize = 22;
pub const CMD_DUPLICATE: usize = 23;
pub const CMD_MIRROR: usize = 24;
pub const CMD_INTERLEAVE: usize = 25;
pub const CMD_TOOL_SET: usize = 26;
pub const CMD_TOOL_ADD: usize = 27;
pub const CMD_TOOL_COPY: usize = 28;
pub const CMD_TOOL_COMMIT: usize = 29;

// Value tokens: 32..48 (integers 0..15)
pub const VAL_BASE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskFamilyType {
    FsmPermutation,       // Meta-train
    FsmRingLattice,       // Meta-val
    FsmHierarchicalTree,  // Meta-test (held-out)
    SymbolicPrimitives,   // Meta-train
    SymbolicCompound,     // Meta-val
    SymbolicCompositional,// Meta-test (held-out)
    ToolWorkflowBasic,    // Meta-train
    ToolWorkflowChained,  // Meta-test (held-out)
}

impl TaskFamilyType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::FsmPermutation => "fsm_permutation_train",
            Self::FsmRingLattice => "fsm_ring_lattice_val",
            Self::FsmHierarchicalTree => "fsm_hierarchical_tree_test",
            Self::SymbolicPrimitives => "symbolic_primitives_train",
            Self::SymbolicCompound => "symbolic_compound_val",
            Self::SymbolicCompositional => "symbolic_compositional_test",
            Self::ToolWorkflowBasic => "tool_workflow_basic_train",
            Self::ToolWorkflowChained => "tool_workflow_chained_test",
        }
    }

    pub fn is_held_out(&self) -> bool {
        matches!(
            self,
            Self::FsmHierarchicalTree | Self::SymbolicCompositional | Self::ToolWorkflowChained
        )
    }
}

pub struct TaskGenerator;

impl TaskGenerator {
    pub fn generate_split(family: TaskFamilyType, seed: u64, n_train: usize, n_val: usize, n_test: usize) -> TaskSplit {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let train_examples = (0..n_train).map(|_| Self::generate_single(&mut rng, family)).collect();
        let val_examples = (0..n_val).map(|_| Self::generate_single(&mut rng, family)).collect();
        let test_examples = (0..n_test).map(|_| Self::generate_single(&mut rng, family)).collect();

        TaskSplit {
            family_name: family.name().to_string(),
            is_held_out: family.is_held_out(),
            train_examples,
            val_examples,
            test_examples,
        }
    }

    fn generate_single(rng: &mut ChaCha8Rng, family: TaskFamilyType) -> TaskExample {
        match family {
            TaskFamilyType::FsmPermutation => Self::generate_fsm_permutation(rng),
            TaskFamilyType::FsmRingLattice => Self::generate_fsm_ring(rng),
            TaskFamilyType::FsmHierarchicalTree => Self::generate_fsm_tree(rng),
            TaskFamilyType::SymbolicPrimitives => Self::generate_symbolic_primitive(rng),
            TaskFamilyType::SymbolicCompound => Self::generate_symbolic_compound(rng),
            TaskFamilyType::SymbolicCompositional => Self::generate_symbolic_compositional(rng),
            TaskFamilyType::ToolWorkflowBasic => Self::generate_tool_basic(rng),
            TaskFamilyType::ToolWorkflowChained => Self::generate_tool_chained(rng),
        }
    }

    // FSM Permutation: 4 states, random transition permutation per symbol
    fn generate_fsm_permutation(rng: &mut ChaCha8Rng) -> TaskExample {
        let n_states = 4;
        let n_syms = 4;
        let seq_len = rng.gen_range(6..=10);

        let mut transitions = vec![vec![0usize; n_syms]; n_states];
        let mut emissions = vec![vec![0usize; n_syms]; n_states];
        for s in 0..n_states {
            for a in 0..n_syms {
                transitions[s][a] = (s + a + 1) % n_states;
                emissions[s][a] = SYMBOL_BASE + ((s * 2 + a) % NUM_SYMBOLS);
            }
        }

        let mut curr_state = 0;
        let mut input_tokens = vec![BOS_TOKEN, STATE_BASE + curr_state, SEP_TOKEN];
        let mut target_tokens = Vec::with_capacity(seq_len + 1);

        for _ in 0..seq_len {
            let a = rng.gen_range(0..n_syms);
            input_tokens.push(SYMBOL_BASE + a);
            target_tokens.push(emissions[curr_state][a]);
            curr_state = transitions[curr_state][a];
        }
        input_tokens.push(SEP_TOKEN);
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // FSM Ring Lattice: ring topology with next/prev transitions
    fn generate_fsm_ring(rng: &mut ChaCha8Rng) -> TaskExample {
        let n_states = 6;
        let seq_len = rng.gen_range(6..=10);
        let mut curr_state = rng.gen_range(0..n_states);
        let mut input_tokens = vec![BOS_TOKEN, STATE_BASE + curr_state, SEP_TOKEN];
        let mut target_tokens = Vec::with_capacity(seq_len + 1);

        for _ in 0..seq_len {
            let dir = rng.gen_range(0..2); // 0 = forward, 1 = backward
            input_tokens.push(SYMBOL_BASE + dir);
            curr_state = if dir == 0 { (curr_state + 1) % n_states } else { (curr_state + n_states - 1) % n_states };
            target_tokens.push(SYMBOL_BASE + curr_state);
        }
        input_tokens.push(SEP_TOKEN);
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // FSM Tree: binary tree with reset transitions (Held-out structure!)
    fn generate_fsm_tree(rng: &mut ChaCha8Rng) -> TaskExample {
        let seq_len = rng.gen_range(6..=10);
        let mut curr_state = 0usize; // root = 0, left = 1, right = 2, leaves = 3..6
        let mut input_tokens = vec![BOS_TOKEN, STATE_BASE + curr_state, SEP_TOKEN];
        let mut target_tokens = Vec::with_capacity(seq_len + 1);

        for _ in 0..seq_len {
            let choice = rng.gen_range(0..2); // 0: left, 1: right
            input_tokens.push(SYMBOL_BASE + choice);
            if curr_state >= 3 {
                curr_state = 0; // reset to root
            } else if curr_state == 0 {
                curr_state = if choice == 0 { 1 } else { 2 };
            } else if curr_state == 1 {
                curr_state = if choice == 0 { 3 } else { 4 };
            } else {
                curr_state = if choice == 0 { 5 } else { 6 };
            }
            target_tokens.push(SYMBOL_BASE + (curr_state % NUM_SYMBOLS));
        }
        input_tokens.push(SEP_TOKEN);
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // Symbolic Primitives: Shift or Duplicate
    fn generate_symbolic_primitive(rng: &mut ChaCha8Rng) -> TaskExample {
        let op = if rng.gen_bool(0.5) { CMD_SHIFT_1 } else { CMD_DUPLICATE };
        let seq_len = rng.gen_range(4..=6);
        let symbols: Vec<usize> = (0..seq_len).map(|_| SYMBOL_BASE + rng.gen_range(0..NUM_SYMBOLS)).collect();

        let mut input_tokens = vec![BOS_TOKEN, op, SEP_TOKEN];
        input_tokens.extend(&symbols);
        input_tokens.push(SEP_TOKEN);

        let mut target_tokens = Vec::new();
        if op == CMD_SHIFT_1 {
            for &s in &symbols {
                let shifted = SYMBOL_BASE + ((s - SYMBOL_BASE + 1) % NUM_SYMBOLS);
                target_tokens.push(shifted);
            }
        } else {
            for &s in &symbols {
                target_tokens.push(s);
                target_tokens.push(s);
            }
        }
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // Symbolic Compound: Reverse or Shift-2
    fn generate_symbolic_compound(rng: &mut ChaCha8Rng) -> TaskExample {
        let op = if rng.gen_bool(0.5) { CMD_REVERSE } else { CMD_SHIFT_2 };
        let seq_len = rng.gen_range(4..=6);
        let symbols: Vec<usize> = (0..seq_len).map(|_| SYMBOL_BASE + rng.gen_range(0..NUM_SYMBOLS)).collect();

        let mut input_tokens = vec![BOS_TOKEN, op, SEP_TOKEN];
        input_tokens.extend(&symbols);
        input_tokens.push(SEP_TOKEN);

        let mut target_tokens = Vec::new();
        if op == CMD_REVERSE {
            let mut rev = symbols.clone();
            rev.reverse();
            target_tokens.extend(rev);
        } else {
            for &s in &symbols {
                let shifted = SYMBOL_BASE + ((s - SYMBOL_BASE + 2) % NUM_SYMBOLS);
                target_tokens.push(shifted);
            }
        }
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // Symbolic Compositional (Held-out): Mirror-Duplicate (x -> [x, rev(x), duplicate(first)])
    fn generate_symbolic_compositional(rng: &mut ChaCha8Rng) -> TaskExample {
        let op = CMD_MIRROR;
        let seq_len = rng.gen_range(3..=5);
        let symbols: Vec<usize> = (0..seq_len).map(|_| SYMBOL_BASE + rng.gen_range(0..NUM_SYMBOLS)).collect();

        let mut input_tokens = vec![BOS_TOKEN, op, SEP_TOKEN];
        input_tokens.extend(&symbols);
        input_tokens.push(SEP_TOKEN);

        let mut target_tokens = Vec::new();
        target_tokens.extend(&symbols);
        let mut rev = symbols.clone();
        rev.reverse();
        target_tokens.extend(rev);
        target_tokens.push(EOS_TOKEN);

        TaskExample { input_tokens, target_tokens }
    }

    // Tool Workflow Basic: Set variable then commit
    fn generate_tool_basic(rng: &mut ChaCha8Rng) -> TaskExample {
        let target_var = rng.gen_range(0..4);
        let target_val = rng.gen_range(1..10);

        let input_tokens = vec![
            BOS_TOKEN,
            CMD_TOOL_SET,
            VAL_BASE + target_var,
            VAL_BASE + target_val,
            SEP_TOKEN,
        ];

        let target_tokens = vec![
            CMD_TOOL_SET,
            VAL_BASE + target_var,
            VAL_BASE + target_val,
            CMD_TOOL_COMMIT,
            EOS_TOKEN,
        ];

        TaskExample { input_tokens, target_tokens }
    }

    // Tool Workflow Chained (Held-out): Add R0, R1 -> R2, then Commit
    fn generate_tool_chained(rng: &mut ChaCha8Rng) -> TaskExample {
        let v0 = rng.gen_range(1..6);
        let v1 = rng.gen_range(1..6);
        let sum = (v0 + v1) % 15;

        let input_tokens = vec![
            BOS_TOKEN,
            CMD_TOOL_ADD,
            VAL_BASE + v0,
            VAL_BASE + v1,
            SEP_TOKEN,
        ];

        let target_tokens = vec![
            CMD_TOOL_SET,
            VAL_BASE + 0,
            VAL_BASE + v0,
            CMD_TOOL_SET,
            VAL_BASE + 1,
            VAL_BASE + v1,
            CMD_TOOL_ADD,
            VAL_BASE + 2, // dst = R2
            VAL_BASE + sum,
            CMD_TOOL_COMMIT,
            EOS_TOKEN,
        ];

        TaskExample { input_tokens, target_tokens }
    }
}

pub struct ImmutableOracle;

impl ImmutableOracle {
    pub fn is_exact_match(predicted: &[usize], target: &[usize]) -> bool {
        if predicted.len() != target.len() {
            return false;
        }
        predicted == target
    }

    pub fn token_accuracy(predicted: &[usize], target: &[usize]) -> f32 {
        if target.is_empty() {
            return 0.0;
        }
        let matches = predicted.iter().zip(target.iter()).filter(|(&p, &t)| p == t).count();
        matches as f32 / target.len() as f32
    }

    /// Primary scoring interface: Exact full sequence match (non-negotiable primary evaluation metric)
    pub fn score_batch(predictions: &[Vec<usize>], targets: &[TaskExample]) -> f32 {
        Self::score_batch_exact(predictions, targets)
    }

    pub fn score_batch_exact(predictions: &[Vec<usize>], targets: &[TaskExample]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        let matches = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, ex)| Self::is_exact_match(pred, &ex.target_tokens))
            .count();
        matches as f32 / targets.len() as f32
    }

    pub fn score_batch_tokens(predictions: &[Vec<usize>], targets: &[TaskExample]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        let sum: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, ex)| Self::token_accuracy(pred, &ex.target_tokens))
            .sum();
        sum / targets.len() as f32
    }

    pub fn score_batch_composite(predictions: &[Vec<usize>], targets: &[TaskExample]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        let token_acc = Self::score_batch_tokens(predictions, targets);
        let exact_acc = Self::score_batch_exact(predictions, targets);
        0.7 * token_acc + 0.3 * exact_acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_generation_deterministic() {
        let split1 = TaskGenerator::generate_split(TaskFamilyType::FsmPermutation, 42, 10, 5, 5);
        let split2 = TaskGenerator::generate_split(TaskFamilyType::FsmPermutation, 42, 10, 5, 5);
        assert_eq!(split1.train_examples.len(), 10);
        assert_eq!(split1.train_examples[0].input_tokens, split2.train_examples[0].input_tokens);
        assert_eq!(split1.train_examples[0].target_tokens, split2.train_examples[0].target_tokens);
    }

    #[test]
    fn test_held_out_flags() {
        assert!(!TaskFamilyType::FsmPermutation.is_held_out());
        assert!(TaskFamilyType::FsmHierarchicalTree.is_held_out());
        assert!(TaskFamilyType::SymbolicCompositional.is_held_out());
        assert!(TaskFamilyType::ToolWorkflowChained.is_held_out());
    }

    #[test]
    fn test_oracle_exact_scoring() {
        let ex = TaskExample {
            input_tokens: vec![1, 4, 3],
            target_tokens: vec![5, 2],
        };
        assert!(ImmutableOracle::is_exact_match(&[5, 2], &ex.target_tokens));
        assert!(!ImmutableOracle::is_exact_match(&[5, 3], &ex.target_tokens));
    }
}
