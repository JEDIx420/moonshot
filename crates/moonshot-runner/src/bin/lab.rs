//! Moonshot Lab: Interactive CLI REPL for token-by-token transformer execution,
//! live sandbox register/memory state inspection, and multi-head attention matrix visualization.

use std::fmt::Write as FmtWrite;
use std::io::{self, BufRead, Write};
use std::process;

use clap::Parser;
use moonshot_core::TaskExample;
use moonshot_model::{ModelConfig, TinyTransformer};
use moonshot_oracle::{
    SandboxState, TaskFamilyType, TaskGenerator,
    BOS_TOKEN, CMD_DUPLICATE, CMD_INTERLEAVE, CMD_MIRROR, CMD_REVERSE, CMD_SHIFT_1, CMD_SHIFT_2,
    CMD_TOOL_ADD, CMD_TOOL_BRANCH, CMD_TOOL_COMMIT, CMD_TOOL_COPY, CMD_TOOL_RECOVER, CMD_TOOL_SET,
    EOS_TOKEN, NUM_SYMBOLS, PAD_TOKEN, SEP_TOKEN, STATE_BASE, SYMBOL_BASE, VAL_BASE, VOCAB_SIZE,
};

/// Command line arguments for Moonshot Lab REPL
#[derive(Parser, Debug)]
#[command(name = "lab", author = "Moonshot Team", version = "0.1.0", about = "Moonshot Lab Interactive CLI REPL")]
pub struct LabCli {
    /// Model size: 77k or 1m
    #[arg(short = 'm', long = "model", default_value = "77k")]
    pub model: String,

    /// Random seed for model initialization and task generation
    #[arg(short = 's', long = "seed", default_value_t = 42)]
    pub seed: u64,

    /// Task family: tool_sandbox, tool_basic, tool_chained, fsm_tree, fsm_perm, fsm_ring, sym_prim, sym_comp, sym_comp_test
    #[arg(short = 'f', long = "family", default_value = "tool_sandbox")]
    pub family: String,

    /// Custom initial prompt tokens (comma or space separated IDs or names)
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// Batch/scripted commands to execute sequentially before or instead of interactive loop (semicolon or newline separated)
    #[arg(short = 'c', long = "commands")]
    pub commands: Option<String>,

    /// Non-interactive batch mode (exit immediately after executing commands or initial setup)
    #[arg(short = 'b', long = "batch", default_value_t = false)]
    pub batch: bool,
}

// ============================================================================
// Token Decoding and ANSI Formatting
// ============================================================================

/// Return a human-readable symbolic name for a given vocabulary token ID.
pub fn token_name(tok: usize) -> String {
    match tok {
        PAD_TOKEN => "PAD".to_string(),
        BOS_TOKEN => "BOS".to_string(),
        EOS_TOKEN => "EOS".to_string(),
        SEP_TOKEN => "SEP".to_string(),
        t if (SYMBOL_BASE..SYMBOL_BASE + NUM_SYMBOLS).contains(&t) => {
            let ch = (b'A' + (t - SYMBOL_BASE) as u8) as char;
            ch.to_string()
        }
        t if (STATE_BASE..STATE_BASE + 8).contains(&t) => {
            format!("S{}", t - STATE_BASE)
        }
        CMD_REVERSE => "REVERSE".to_string(),
        CMD_SHIFT_1 => "SHIFT_1".to_string(),
        CMD_SHIFT_2 => "SHIFT_2".to_string(),
        CMD_DUPLICATE => "DUPLICATE".to_string(),
        CMD_MIRROR => "MIRROR".to_string(),
        CMD_INTERLEAVE => "INTERLEAVE".to_string(),
        CMD_TOOL_SET => "SET".to_string(),
        CMD_TOOL_ADD => "ADD".to_string(),
        CMD_TOOL_COPY => "COPY".to_string(),
        CMD_TOOL_COMMIT => "COMMIT".to_string(),
        CMD_TOOL_BRANCH => "BRANCH".to_string(),
        CMD_TOOL_RECOVER => "RECOVER".to_string(),
        t if (VAL_BASE..VAL_BASE + 16).contains(&t) => {
            let val = t - VAL_BASE;
            format!("V{}", val)
        }
        t if t < VOCAB_SIZE => format!("T{}", t),
        _ => format!("?{}", tok),
    }
}

/// Parse a token string (either integer ID or name like "BOS", "SET", "V0", "R0", "A") into a token ID.
pub fn parse_token_str(s: &str) -> Option<usize> {
    let clean = s.trim().to_uppercase();
    if let Ok(id) = clean.parse::<usize>() {
        if id < VOCAB_SIZE {
            return Some(id);
        }
    }
    match clean.as_str() {
        "PAD" => Some(PAD_TOKEN),
        "BOS" => Some(BOS_TOKEN),
        "EOS" => Some(EOS_TOKEN),
        "SEP" => Some(SEP_TOKEN),
        "A" => Some(SYMBOL_BASE),
        "B" => Some(SYMBOL_BASE + 1),
        "C" => Some(SYMBOL_BASE + 2),
        "D" => Some(SYMBOL_BASE + 3),
        "E" => Some(SYMBOL_BASE + 4),
        "F" => Some(SYMBOL_BASE + 5),
        "G" => Some(SYMBOL_BASE + 6),
        "H" => Some(SYMBOL_BASE + 7),
        "REVERSE" => Some(CMD_REVERSE),
        "SHIFT_1" => Some(CMD_SHIFT_1),
        "SHIFT_2" => Some(CMD_SHIFT_2),
        "DUPLICATE" => Some(CMD_DUPLICATE),
        "MIRROR" => Some(CMD_MIRROR),
        "INTERLEAVE" => Some(CMD_INTERLEAVE),
        "SET" => Some(CMD_TOOL_SET),
        "ADD" => Some(CMD_TOOL_ADD),
        "COPY" => Some(CMD_TOOL_COPY),
        "COMMIT" => Some(CMD_TOOL_COMMIT),
        "BRANCH" => Some(CMD_TOOL_BRANCH),
        "RECOVER" => Some(CMD_TOOL_RECOVER),
        s if s.starts_with('S') => {
            let num: usize = s[1..].parse().ok()?;
            if num < 8 {
                Some(STATE_BASE + num)
            } else {
                None
            }
        }
        s if s.starts_with('V') => {
            let num: usize = s[1..].parse().ok()?;
            if num < 16 {
                Some(VAL_BASE + num)
            } else {
                None
            }
        }
        s if s.starts_with('R') => {
            // Register tokens correspond to VAL_BASE + 0..3
            let num: usize = s[1..].parse().ok()?;
            if num < 4 {
                Some(VAL_BASE + num)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Format a token with ANSI terminal color styling.
pub fn format_token_ansi(tok: usize, is_generated: bool) -> String {
    let name = token_name(tok);
    let raw = format!("{}[{}]", tok, name);

    match tok {
        BOS_TOKEN => format!("\x1b[1;32m{}\x1b[0m", raw), // Bold Green
        EOS_TOKEN => format!("\x1b[1;31m{}\x1b[0m", raw), // Bold Red
        SEP_TOKEN => format!("\x1b[1;33m{}\x1b[0m", raw), // Bold Yellow
        PAD_TOKEN => format!("\x1b[2;37m{}\x1b[0m", raw), // Dim White
        t if (CMD_REVERSE..=CMD_TOOL_RECOVER).contains(&t) => {
            // Commands: Bold Magenta
            if is_generated {
                format!("\x1b[1;4;35m{}\x1b[0m", raw)
            } else {
                format!("\x1b[1;35m{}\x1b[0m", raw)
            }
        }
        t if (VAL_BASE..VAL_BASE + 16).contains(&t) => {
            // Values / Registers: Cyan
            if is_generated {
                format!("\x1b[1;4;36m{}\x1b[0m", raw)
            } else {
                format!("\x1b[36m{}\x1b[0m", raw)
            }
        }
        t if (STATE_BASE..STATE_BASE + 8).contains(&t) => {
            // States: Blue
            format!("\x1b[1;34m{}\x1b[0m", raw)
        }
        t if (SYMBOL_BASE..SYMBOL_BASE + NUM_SYMBOLS).contains(&t) => {
            // Symbols: Green
            format!("\x1b[32m{}\x1b[0m", raw)
        }
        _ => {
            if is_generated {
                format!("\x1b[4m{}\x1b[0m", raw)
            } else {
                raw
            }
        }
    }
}

/// Parse task family string to `TaskFamilyType`.
pub fn parse_task_family(s: &str) -> TaskFamilyType {
    let lower = s.to_lowercase().replace('-', "_");
    match lower.as_str() {
        "tool_sandbox" | "tool_workflow_sandbox" | "sandbox" => TaskFamilyType::ToolWorkflowSandbox,
        "tool_basic" | "tool_workflow_basic" => TaskFamilyType::ToolWorkflowBasic,
        "tool_chained" | "tool_workflow_chained" => TaskFamilyType::ToolWorkflowChained,
        "fsm_tree" | "fsm_hierarchical_tree" | "tree" => TaskFamilyType::FsmHierarchicalTree,
        "fsm_perm" | "fsm_permutation" | "perm" => TaskFamilyType::FsmPermutation,
        "fsm_ring" | "fsm_ring_lattice" | "ring" => TaskFamilyType::FsmRingLattice,
        "sym_prim" | "symbolic_primitives" | "prim" => TaskFamilyType::SymbolicPrimitives,
        "sym_comp" | "symbolic_compound" | "comp" => TaskFamilyType::SymbolicCompound,
        "sym_comp_test" | "symbolic_compositional" | "compositional" => {
            TaskFamilyType::SymbolicCompositional
        }
        _ => TaskFamilyType::ToolWorkflowSandbox,
    }
}

// ============================================================================
// Model Forward Pass with Multi-Head Attention Extraction
// ============================================================================

/// Perform a forward pass through `TinyTransformer` while collecting attention matrices
/// `[n_heads * seq_len * seq_len]` from every transformer block.
pub fn forward_with_attention(
    model: &TinyTransformer,
    tokens: &[usize],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    model.forward_with_attention(tokens)
}

/// Compute top-K predictions for the last token position from logits.
pub fn top_k_predictions(
    logits: &[f32],
    vocab_size: usize,
    seq_len: usize,
    k: usize,
) -> Vec<(usize, f32, f32)> {
    if seq_len == 0 {
        return Vec::new();
    }
    let last_logits = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

    // Compute numerically stable softmax
    let mut max_l = f32::NEG_INFINITY;
    for &l in last_logits {
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum_exp = 0.0;
    let mut probs = vec![0.0; vocab_size];
    for (i, &l) in last_logits.iter().enumerate() {
        let e = (l - max_l).exp();
        probs[i] = e;
        sum_exp += e;
    }
    let denom = sum_exp.max(1e-9);
    for p in probs.iter_mut() {
        *p /= denom;
    }

    let mut indexed: Vec<(usize, f32, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(tok, &l)| (tok, l, probs[tok]))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ============================================================================
// ANSI Card and Visual Renderers
// ============================================================================

/// Render a Unicode bordered box containing live Sandbox register values.
pub fn render_registers_box(registers: &[i32; 4]) -> String {
    let mut out = String::new();
    writeln!(out, "  \x1b[1;34m┌────────────────────── Registers (R0..R3) ──────────────────────┐\x1b[0m").unwrap();
    write!(out, "  \x1b[1;34m│\x1b[0m").unwrap();
    for (i, &val) in registers.iter().enumerate() {
        write!(
            out,
            "   \x1b[1;36mR{}:\x1b[0m \x1b[1;37m{:2}\x1b[0m (0x{:X})   \x1b[1;34m│\x1b[0m",
            i, val, val
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  \x1b[1;34m└────────────────────────────────────────────────────────────────┘\x1b[0m").unwrap();
    out
}

/// Render a Unicode bordered box containing live Sandbox memory values.
pub fn render_memory_box(memory: &[i32; 4]) -> String {
    let mut out = String::new();
    writeln!(out, "  \x1b[1;33m┌─────────────────────── Memory (M0..M3) ────────────────────────┐\x1b[0m").unwrap();
    write!(out, "  \x1b[1;33m│\x1b[0m").unwrap();
    for (i, &val) in memory.iter().enumerate() {
        write!(
            out,
            "   \x1b[1;33mM{}:\x1b[0m \x1b[1;37m{:2}\x1b[0m (0x{:X})   \x1b[1;33m│\x1b[0m",
            i, val, val
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  \x1b[1;33m└────────────────────────────────────────────────────────────────┘\x1b[0m").unwrap();
    out
}

/// Render sandbox status card (committed, error, steps executed).
pub fn render_sandbox_status(
    state: &SandboxState,
    steps_executed: usize,
    failure_reason: Option<&str>,
) -> String {
    let mut out = String::new();
    let commit_badge = if state.committed {
        "\x1b[1;42;37m COMMITTED \x1b[0m"
    } else {
        "\x1b[1;43;30m UNCOMMITTED \x1b[0m"
    };

    let error_badge = if state.has_error || failure_reason.is_some() {
        format!(
            "\x1b[1;41;37m ERROR \x1b[0m \x1b[31m({})\x1b[0m",
            failure_reason.unwrap_or("has_error flagged")
        )
    } else {
        "\x1b[1;32mOK (no fault)\x1b[0m".to_string()
    };

    writeln!(
        out,
        "  \x1b[1mSandbox State:\x1b[0m {} | Steps: \x1b[1m{}/32\x1b[0m | Health: {}",
        commit_badge, steps_executed, error_badge
    )
    .unwrap();
    out
}

/// Map an attention weight (0.0 .. 1.0) to character density block and ANSI color.
pub fn format_attention_cell(w: f32) -> String {
    if w >= 0.70 {
        "\x1b[1;31m█\x1b[0m".to_string() // Red
    } else if w >= 0.40 {
        "\x1b[1;33m▓\x1b[0m".to_string() // Yellow
    } else if w >= 0.20 {
        "\x1b[36m▒\x1b[0m".to_string() // Cyan
    } else if w >= 0.05 {
        "\x1b[34m░\x1b[0m".to_string() // Blue
    } else {
        " ".to_string()
    }
}

/// Render a 2D attention heatmap matrix for a specific layer and head.
pub fn render_attention_heatmap(
    attn_weights: &[f32],
    tokens: &[usize],
    layer: usize,
    head: usize,
    _n_heads: usize,
) -> String {
    let seq_len = tokens.len();
    let mut out = String::new();

    writeln!(
        out,
        "  \x1b[1;35m--- Attention Heatmap [Layer {}, Head {}] (Seq Len: {}) ---\x1b[0m",
        layer, head, seq_len
    )
    .unwrap();

    if seq_len == 0 {
        writeln!(out, "  (No tokens in sequence)").unwrap();
        return out;
    }

    // Header column labels
    write!(out, "       ").unwrap();
    for &tok in tokens.iter().take(seq_len) {
        let name = token_name(tok);
        let short_name = if name.len() > 4 { &name[..4] } else { &name };
        write!(out, " {:>4}", short_name).unwrap();
    }
    writeln!(out).unwrap();

    let head_offset = head * seq_len * seq_len;
    for (i, &tok) in tokens.iter().enumerate().take(seq_len) {
        let q_name = token_name(tok);
        let short_q = if q_name.len() > 4 { &q_name[..4] } else { &q_name };
        write!(out, "  {:>2} {:>4}", i, short_q).unwrap();

        for j in 0..seq_len {
            if j <= i {
                let idx = head_offset + i * seq_len + j;
                let w = if idx < attn_weights.len() {
                    attn_weights[idx]
                } else {
                    0.0
                };
                let cell = format_attention_cell(w);
                write!(out, "    {}", cell).unwrap();
            } else {
                write!(out, "     ").unwrap();
            }
        }
        writeln!(out).unwrap();
    }

    writeln!(
        out,
        "  Density: \x1b[1;31m█\x1b[0m >=0.70  \x1b[1;33m▓\x1b[0m >=0.40  \x1b[36m▒\x1b[0m >=0.20  \x1b[34m░\x1b[0m >=0.05  [ ] <0.05"
    )
    .unwrap();
    out
}

/// Render an ASCII bar chart of the attention distribution from the last query token to all keys.
pub fn render_attention_bar_chart(
    attn_weights: &[f32],
    tokens: &[usize],
    layer: usize,
    head: usize,
) -> String {
    let seq_len = tokens.len();
    let mut out = String::new();
    if seq_len == 0 {
        return out;
    }
    let query_idx = seq_len - 1;
    let query_tok = tokens[query_idx];
    let query_name = token_name(query_tok);

    writeln!(
        out,
        "  \x1b[1mQuery Pos {} [{}] Attention Distribution (Layer {}, Head {}):\x1b[0m",
        query_idx, query_name, layer, head
    )
    .unwrap();

    let head_offset = head * seq_len * seq_len;
    let max_bar_len = 30;

    for (j, &tok) in tokens.iter().enumerate().take(seq_len) {
        let idx = head_offset + query_idx * seq_len + j;
        let w = if idx < attn_weights.len() {
            attn_weights[idx]
        } else {
            0.0
        };
        let bar_len = ((w * max_bar_len as f32).round() as usize).min(max_bar_len);
        let bar: String = "█".repeat(bar_len);
        let k_name = token_name(tok);

        let color_code = if w > 0.4 {
            "\x1b[1;31m"
        } else if w > 0.2 {
            "\x1b[1;33m"
        } else if w > 0.05 {
            "\x1b[36m"
        } else {
            "\x1b[34m"
        };

        writeln!(
            out,
            "    Key [{:2}] {:>7}  ({:5.3})  {}{}\x1b[0m",
            j, k_name, w, color_code, bar
        )
        .unwrap();
    }
    out
}

// ============================================================================
// Lab Session State
// ============================================================================

pub struct LabSession {
    pub model: TinyTransformer,
    pub model_name: String,
    pub seed: u64,
    pub family_type: TaskFamilyType,
    pub task_example: TaskExample,
    pub prompt_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub all_tokens: Vec<usize>,
    pub last_logits: Vec<f32>,
    pub last_attention: Vec<Vec<f32>>,
    pub sandbox_state: SandboxState,
    pub sandbox_steps: usize,
    pub sandbox_error: Option<String>,
    pub sandbox_legal: bool,
    pub is_eos_reached: bool,
}

impl LabSession {
    pub fn new(model_name: &str, seed: u64, family_str: &str, custom_prompt: Option<String>) -> Self {
        let config = if model_name.to_lowercase().contains("1m") {
            ModelConfig::new_1m()
        } else {
            ModelConfig::new_77k()
        };

        let model = TinyTransformer::new(config, seed);
        let family_type = parse_task_family(family_str);

        let split = TaskGenerator::generate_split(family_type, seed, 1, 1, 1);
        let task_example = split
            .train_examples
            .into_iter()
            .next()
            .unwrap_or_else(|| TaskExample {
                input_tokens: vec![BOS_TOKEN, CMD_TOOL_SET, VAL_BASE + 2, VAL_BASE + 3, SEP_TOKEN],
                target_tokens: vec![
                    CMD_TOOL_SET,
                    VAL_BASE,
                    VAL_BASE + 2,
                    CMD_TOOL_COMMIT,
                    EOS_TOKEN,
                ],
            });

        let prompt_tokens = if let Some(p) = custom_prompt {
            Self::parse_prompt_string(&p)
        } else {
            task_example.input_tokens.clone()
        };

        let all_tokens = prompt_tokens.clone();
        let (logits, attn) = forward_with_attention(&model, &all_tokens);

        let mut session = Self {
            model,
            model_name: model_name.to_string(),
            seed,
            family_type,
            task_example,
            prompt_tokens,
            generated_tokens: Vec::new(),
            all_tokens,
            last_logits: logits,
            last_attention: attn,
            sandbox_state: SandboxState::new(),
            sandbox_steps: 0,
            sandbox_error: None,
            sandbox_legal: false,
            is_eos_reached: false,
        };

        session.update_sandbox_state();
        session
    }

    fn parse_prompt_string(s: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        for part in s.split(|c: char| c == ',' || c.is_whitespace()) {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(tok) = parse_token_str(trimmed) {
                tokens.push(tok);
            }
        }
        if tokens.is_empty() {
            vec![BOS_TOKEN, SEP_TOKEN]
        } else {
            tokens
        }
    }

    /// Reset session with optionally new family and seed.
    pub fn reset(&mut self, family_opt: Option<&str>, seed_opt: Option<u64>) {
        if let Some(s) = seed_opt {
            self.seed = s;
            let config = if self.model_name.to_lowercase().contains("1m") {
                ModelConfig::new_1m()
            } else {
                ModelConfig::new_77k()
            };
            self.model = TinyTransformer::new(config, self.seed);
        }

        if let Some(f) = family_opt {
            self.family_type = parse_task_family(f);
        }

        let split = TaskGenerator::generate_split(self.family_type, self.seed, 1, 1, 1);
        self.task_example = split
            .train_examples
            .into_iter()
            .next()
            .unwrap_or_else(|| TaskExample {
                input_tokens: vec![BOS_TOKEN, SEP_TOKEN],
                target_tokens: vec![EOS_TOKEN],
            });

        self.prompt_tokens = self.task_example.input_tokens.clone();
        self.generated_tokens.clear();
        self.all_tokens = self.prompt_tokens.clone();
        self.is_eos_reached = false;

        let (logits, attn) = forward_with_attention(&self.model, &self.all_tokens);
        self.last_logits = logits;
        self.last_attention = attn;

        self.update_sandbox_state();
    }

    /// Step generation by `count` tokens.
    pub fn step(&mut self, count: usize) {
        for _ in 0..count {
            if self.is_eos_reached {
                println!("  \x1b[33m[Generation halted: EOS_TOKEN (2) already emitted]\x1b[0m");
                break;
            }
            if self.all_tokens.len() >= self.model.config.max_seq_len {
                println!(
                    "  \x1b[33m[Generation halted: reached max sequence length ({})]\x1b[0m",
                    self.model.config.max_seq_len
                );
                break;
            }

            // Current forward pass logits
            let seq_len = self.all_tokens.len().min(self.model.config.max_seq_len);
            let top = top_k_predictions(
                &self.last_logits,
                self.model.config.vocab_size,
                seq_len,
                3,
            );

            if top.is_empty() {
                break;
            }

            let next_tok = top[0].0;
            let prob = top[0].2;
            let tok_str = format_token_ansi(next_tok, true);

            self.generated_tokens.push(next_tok);
            self.all_tokens.push(next_tok);

            if next_tok == EOS_TOKEN {
                self.is_eos_reached = true;
            }

            // Re-run forward pass to update logits and attention matrices
            let (logits, attn) = forward_with_attention(&self.model, &self.all_tokens);
            self.last_logits = logits;
            self.last_attention = attn;

            self.update_sandbox_state();

            println!(
                "  Step {:2} -> Generated: {} (p={:5.3}) | Total Seq Len: {}",
                self.generated_tokens.len(),
                tok_str,
                prob,
                self.all_tokens.len()
            );

            if top.len() > 1 {
                let mut alt = String::new();
                for (t, _, p) in top.iter().skip(1) {
                    write!(alt, " {} (p={:5.3});", format_token_ansi(*t, false), p).unwrap();
                }
                println!("          Candidates:{}", alt);
            }

            if self.is_eos_reached {
                println!("  \x1b[1;32m✓ Generation sequence complete with EOS_TOKEN\x1b[0m");
                break;
            }
        }
    }

    /// Run generation to completion (EOS or max sequence length).
    pub fn run(&mut self) {
        let max_steps = self.model.config.max_seq_len.saturating_sub(self.all_tokens.len());
        println!(
            "  \x1b[1;36mRunning generation (up to {} steps)...\x1b[0m",
            max_steps
        );
        self.step(max_steps);
    }

    /// Update sandbox state based on currently generated tokens or target commands.
    pub fn update_sandbox_state(&mut self) {
        // If we have generated tokens, execute them in sandbox
        let execution_tokens: Vec<usize> = if !self.generated_tokens.is_empty() {
            self.generated_tokens.clone()
        } else {
            // If prompt contains tool commands, find the first tool command and execute from there
            if let Some(pos) = self.prompt_tokens.iter().position(|&t| (CMD_TOOL_SET..=CMD_TOOL_RECOVER).contains(&t)) {
                self.prompt_tokens[pos..].to_vec()
            } else if let Some(sep_pos) = self.prompt_tokens.iter().rposition(|&t| t == SEP_TOKEN) {
                self.prompt_tokens[sep_pos + 1..].to_vec()
            } else {
                Vec::new()
            }
        };

        if execution_tokens.is_empty() {
            self.sandbox_state = SandboxState::new();
            self.sandbox_steps = 0;
            self.sandbox_error = None;
            self.sandbox_legal = false;
        } else {
            // Truncate at SEP_TOKEN if present so trailing delimiters don't trigger unknown token errors
            let valid_tokens = if let Some(sep_idx) = execution_tokens.iter().position(|&t| t == SEP_TOKEN) {
                &execution_tokens[..sep_idx]
            } else {
                &execution_tokens[..]
            };

            let res = SandboxState::execute_tokens(valid_tokens);
            self.sandbox_state = res.final_state;
            self.sandbox_steps = res.steps_executed;
            self.sandbox_error = res.failure_reason;
            self.sandbox_legal = res.is_legal_trace;
        }
    }

    /// Display formatted tokens.
    pub fn show_tokens(&self) {
        println!("  \x1b[1;34m=== Current Token Sequence ({} tokens) ===\x1b[0m", self.all_tokens.len());
        print!("  \x1b[1mPrompt:\x1b[0m ");
        for (i, &tok) in self.prompt_tokens.iter().enumerate() {
            print!("{} ", format_token_ansi(tok, false));
            if i % 8 == 7 && i + 1 < self.prompt_tokens.len() {
                print!("\n          ");
            }
        }
        println!();

        if !self.generated_tokens.is_empty() {
            print!("  \x1b[1mGenerated:\x1b[0m ");
            for (i, &tok) in self.generated_tokens.iter().enumerate() {
                print!("{} ", format_token_ansi(tok, true));
                if i % 8 == 7 && i + 1 < self.generated_tokens.len() {
                    print!("\n             ");
                }
            }
            println!();
        } else {
            println!("  \x1b[2m(No tokens generated yet — use 'step' or 'run')\x1b[0m");
        }
        println!();
    }

    /// Display live registers.
    pub fn show_registers(&self) {
        print!("{}", render_registers_box(&self.sandbox_state.registers));
    }

    /// Display live memory.
    pub fn show_memory(&self) {
        print!("{}", render_memory_box(&self.sandbox_state.memory));
    }

    /// Display attention matrix or distribution.
    pub fn show_attention(&self, layer: usize, head: usize) {
        let n_layers = self.model.config.n_layers;
        let n_heads = self.model.config.n_heads;

        if layer >= n_layers {
            println!(
                "  \x1b[31mInvalid layer {} (model has {} layers: 0..{})\x1b[0m",
                layer,
                n_layers,
                n_layers - 1
            );
            return;
        }
        if head >= n_heads {
            println!(
                "  \x1b[31mInvalid head {} (model has {} heads: 0..{})\x1b[0m",
                head,
                n_heads,
                n_heads - 1
            );
            return;
        }

        let attn_layer = &self.last_attention[layer];
        print!(
            "{}",
            render_attention_heatmap(attn_layer, &self.all_tokens, layer, head, n_heads)
        );
        print!(
            "{}",
            render_attention_bar_chart(attn_layer, &self.all_tokens, layer, head)
        );
    }

    /// Full dashboard inspection.
    pub fn show_inspect(&self) {
        println!("  \x1b[1;36m╔══════════════════════════════════════════════════════════════════╗\x1b[0m");
        println!("  \x1b[1;36m║                  MOONSHOT LAB INSPECT DASHBOARD                  ║\x1b[0m");
        println!("  \x1b[1;36m╚══════════════════════════════════════════════════════════════════╝\x1b[0m");
        println!(
            "  \x1b[1mModel:\x1b[0m {} (L: {}, H: {}, D: {}, Params: {}) | \x1b[1mSeed:\x1b[0m {} | \x1b[1mFamily:\x1b[0m {}",
            self.model_name,
            self.model.config.n_layers,
            self.model.config.n_heads,
            self.model.config.d_model,
            self.model.total_parameters(),
            self.seed,
            self.family_type.name()
        );
        println!(
            "  \x1b[1mTokens:\x1b[0m {} total ({} prompt, {} generated) | \x1b[1mStatus:\x1b[0m {}",
            self.all_tokens.len(),
            self.prompt_tokens.len(),
            self.generated_tokens.len(),
            if self.is_eos_reached {
                "\x1b[1;32mEOS_REACHED\x1b[0m"
            } else {
                "\x1b[1;33mACTIVE\x1b[0m"
            }
        );
        println!();

        self.show_tokens();
        self.show_registers();
        self.show_memory();

        print!(
            "{}",
            render_sandbox_status(
                &self.sandbox_state,
                self.sandbox_steps,
                self.sandbox_error.as_deref()
            )
        );
        println!();

        // Show attention for layer 0, head 0
        if !self.last_attention.is_empty() {
            let last_layer = self.model.config.n_layers - 1;
            println!("  \x1b[1;35m--- Attention Preview (Last Layer {}, Head 0) ---\x1b[0m", last_layer);
            print!(
                "{}",
                render_attention_bar_chart(
                    &self.last_attention[last_layer],
                    &self.all_tokens,
                    last_layer,
                    0
                )
            );
        }

        // Show next token prediction distribution
        let seq_len = self.all_tokens.len().min(self.model.config.max_seq_len);
        let top = top_k_predictions(
            &self.last_logits,
            self.model.config.vocab_size,
            seq_len,
            5,
        );
        println!("  \x1b[1;32m--- Next Token Probabilities (Top 5) ---\x1b[0m");
        for (rank, (tok, logit, prob)) in top.iter().enumerate() {
            println!(
                "    #{}: {} (id: {:2}) -> Logit: {:6.2}, Prob: {:5.1}%",
                rank + 1,
                format_token_ansi(*tok, false),
                tok,
                logit,
                prob * 100.0
            );
        }
        println!();
    }
}

// ============================================================================
// Command Parser and Dispatcher
// ============================================================================

/// Display the REPL command help menu.
pub fn print_help() {
    println!("\x1b[1;36m=== Moonshot Lab REPL Commands ===\x1b[0m");
    println!("  \x1b[1;32mstep [N]\x1b[0m             - Step generation forward by 1 (or N) token(s) (alias: 's')");
    println!("  \x1b[1;32mrun\x1b[0m                  - Run generation until EOS or max sequence length (alias: 'r')");
    println!("  \x1b[1;32mtokens\x1b[0m               - Display current token sequence with ANSI syntax highlighting (alias: 'tok')");
    println!("  \x1b[1;32mregs\x1b[0m                 - Display live SandboxState registers [R0, R1, R2, R3] (alias: 'reg')");
    println!("  \x1b[1;32mmem\x1b[0m                  - Display live SandboxState memory values [M0, M1, M2, M3]");
    println!("  \x1b[1;32mattn [layer] [head]\x1b[0m  - Display attention matrix heatmap and bar distribution (alias: 'a')");
    println!("  \x1b[1;32minspect\x1b[0m              - Full dashboard view (tokens, registers, memory, attention) (alias: 'i')");
    println!("  \x1b[1;32mreset [family] [seed]\x1b[0m- Reset model and sandbox state to initial prompt");
    println!("  \x1b[1;32mtask [family] [seed]\x1b[0m - Change task family (tool_sandbox, tool_basic, fsm_tree, etc.)");
    println!("  \x1b[1;32mprompt <t1> <t2> ...\x1b[0m - Set custom prompt tokens");
    println!("  \x1b[1;32mhelp\x1b[0m                 - Display this command menu (alias: 'h', '?')");
    println!("  \x1b[1;32mquit\x1b[0m / \x1b[1;32mexit\x1b[0m          - Exit the REPL cleanly (alias: 'q')");
    println!();
}

/// Execute a single command line against the current session. Returns false if REPL should exit.
pub fn execute_command(line: &str, session: &mut LabSession) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        // By default on empty Enter: advance by 1 step if not at EOS
        if !session.is_eos_reached && session.all_tokens.len() < session.model.config.max_seq_len {
            session.step(1);
        }
        return true;
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "step" | "s" => {
            let count = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(1)
            } else {
                1
            };
            session.step(count);
        }
        "run" | "r" => {
            session.run();
        }
        "tokens" | "tok" => {
            session.show_tokens();
        }
        "regs" | "reg" => {
            session.show_registers();
        }
        "mem" => {
            session.show_memory();
        }
        "attn" | "a" => {
            let layer = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(0)
            } else {
                0
            };
            let head = if parts.len() > 2 {
                parts[2].parse::<usize>().unwrap_or(0)
            } else {
                0
            };
            session.show_attention(layer, head);
        }
        "inspect" | "i" => {
            session.show_inspect();
        }
        "reset" => {
            let family_arg = parts.get(1).copied();
            let seed_arg = parts.get(2).and_then(|s| s.parse::<u64>().ok());
            session.reset(family_arg, seed_arg);
            println!(
                "  \x1b[32mReset complete. Task family: {}, Seed: {}\x1b[0m",
                session.family_type.name(),
                session.seed
            );
            session.show_tokens();
        }
        "task" | "t" => {
            if parts.len() > 1 {
                let family_arg = parts[1];
                let seed_arg = parts.get(2).and_then(|s| s.parse::<u64>().ok());
                session.reset(Some(family_arg), seed_arg);
                println!(
                    "  \x1b[32mTask loaded: {}, Seed: {}\x1b[0m",
                    session.family_type.name(),
                    session.seed
                );
                session.show_tokens();
            } else {
                println!(
                    "  Current task: {} (seed: {}). Available: tool_sandbox, tool_basic, tool_chained, fsm_tree, fsm_perm, fsm_ring, sym_prim, sym_comp, sym_comp_test",
                    session.family_type.name(),
                    session.seed
                );
            }
        }
        "prompt" | "p" => {
            if parts.len() > 1 {
                let rest = parts[1..].join(" ");
                let new_prompt = LabSession::parse_prompt_string(&rest);
                session.prompt_tokens = new_prompt.clone();
                session.generated_tokens.clear();
                session.all_tokens = new_prompt;
                session.is_eos_reached = false;
                let (logits, attn) = forward_with_attention(&session.model, &session.all_tokens);
                session.last_logits = logits;
                session.last_attention = attn;
                session.update_sandbox_state();
                println!("  \x1b[32mCustom prompt loaded ({} tokens):\x1b[0m", session.all_tokens.len());
                session.show_tokens();
            } else {
                println!("  Usage: prompt <tok1> <tok2> ... (e.g. 'prompt BOS SET 2 3 SEP')");
            }
        }
        "status" => {
            println!(
                "  Model: {} | Family: {} | Seed: {} | SeqLen: {}/{} | Generated: {} | EOS: {}",
                session.model_name,
                session.family_type.name(),
                session.seed,
                session.all_tokens.len(),
                session.model.config.max_seq_len,
                session.generated_tokens.len(),
                session.is_eos_reached
            );
        }
        "help" | "h" | "?" => {
            print_help();
        }
        "quit" | "exit" | "q" => {
            println!("  \x1b[32mExiting Moonshot Lab. Goodbye!\x1b[0m");
            return false;
        }
        unknown => {
            println!(
                "  \x1b[31mUnknown command: '{}'. Type 'help' for available commands.\x1b[0m",
                unknown
            );
        }
    }
    true
}

// ============================================================================
// Main REPL Entrypoint
// ============================================================================

fn main() {
    let args = LabCli::parse();

    println!("\x1b[1;36m╔══════════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[1;36m║               MOONSHOT LAB INTERACTIVE CLI REPL                  ║\x1b[0m");
    println!("\x1b[1;36m║  Token Stepping • Attention Heatmaps • Live Sandbox Inspector    ║\x1b[0m");
    println!("\x1b[1;36m╚══════════════════════════════════════════════════════════════════╝\x1b[0m");

    let mut session = LabSession::new(
        &args.model,
        args.seed,
        &args.family,
        args.prompt,
    );

    println!(
        "  Loaded model \x1b[1m{}\x1b[0m (seed: {}, params: {})",
        session.model_name,
        session.seed,
        session.model.total_parameters()
    );
    println!(
        "  Task Family: \x1b[1;33m{}\x1b[0m | Prompt tokens: \x1b[1;32m{}\x1b[0m",
        session.family_type.name(),
        session.prompt_tokens.len()
    );
    session.show_tokens();
    println!("  Type \x1b[1;32mhelp\x1b[0m for commands, \x1b[1;32mstep\x1b[0m to advance, or \x1b[1;32mquit\x1b[0m to exit.\n");

    // Execute batch commands if provided via CLI argument
    if let Some(cmd_str) = args.commands {
        for cmd in cmd_str.split([';', '\n']) {
            let trimmed = cmd.trim();
            if !trimmed.is_empty() {
                println!("\x1b[1;34m[batch]\x1b[0m > {}", trimmed);
                if !execute_command(trimmed, &mut session) {
                    process::exit(0);
                }
            }
        }
        if args.batch {
            process::exit(0);
        }
    }

    if args.batch {
        process::exit(0);
    }

    // Standard input loop (handles both interactive terminal and piped non-interactive input)
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line_buf = String::new();

    loop {
        print!("\x1b[1;36mmoonshot-lab>\x1b[0m ");
        io::stdout().flush().unwrap_or(());

        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => {
                // EOF reached (e.g. piped input ended)
                println!("\n  \x1b[2m[EOF reached — exiting]\x1b[0m");
                break;
            }
            Ok(_) => {
                let trimmed = line_buf.trim();
                // Echo piped input if not in a TTY so scripted traces are transparent
                if !atty_is_terminal() && !trimmed.is_empty() {
                    println!("{}", trimmed);
                }
                if !execute_command(trimmed, &mut session) {
                    break;
                }
            }
            Err(err) => {
                eprintln!("  Error reading input: {}", err);
                break;
            }
        }
    }

    process::exit(0);
}

/// Simple heuristic to check if stdout is an interactive terminal
fn atty_is_terminal() -> bool {
    // If not a terminal or unsure, don't fail; check basic term env
    std::env::var_os("TERM").is_some()
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_name_and_parsing() {
        assert_eq!(token_name(BOS_TOKEN), "BOS");
        assert_eq!(token_name(EOS_TOKEN), "EOS");
        assert_eq!(token_name(SEP_TOKEN), "SEP");
        assert_eq!(token_name(CMD_TOOL_SET), "SET");
        assert_eq!(token_name(CMD_TOOL_ADD), "ADD");
        assert_eq!(token_name(CMD_TOOL_COPY), "COPY");
        assert_eq!(token_name(CMD_TOOL_COMMIT), "COMMIT");
        assert_eq!(token_name(VAL_BASE), "V0");

        assert_eq!(parse_token_str("BOS"), Some(BOS_TOKEN));
        assert_eq!(parse_token_str("eos"), Some(EOS_TOKEN));
        assert_eq!(parse_token_str("SET"), Some(CMD_TOOL_SET));
        assert_eq!(parse_token_str("V0"), Some(VAL_BASE));
        assert_eq!(parse_token_str("R1"), Some(VAL_BASE + 1));
        assert_eq!(parse_token_str("2"), Some(2));
    }

    #[test]
    fn test_forward_with_attention_matches_forward() {
        let config = ModelConfig::new_77k();
        let model = TinyTransformer::new(config, 123);
        let tokens = vec![BOS_TOKEN, CMD_TOOL_SET, VAL_BASE + 2, SEP_TOKEN];

        let regular_logits = model.forward(&tokens);
        let (attn_logits, attn_weights) = forward_with_attention(&model, &tokens);

        assert_eq!(regular_logits.len(), attn_logits.len());
        for (a, b) in regular_logits.iter().zip(attn_logits.iter()) {
            assert!((a - b).abs() < 1e-6, "Logits must match regular forward pass");
        }

        assert_eq!(attn_weights.len(), model.config.n_layers);
        let seq_len = tokens.len();
        let expected_size = model.config.n_heads * seq_len * seq_len;
        for w in &attn_weights {
            assert_eq!(w.len(), expected_size);
        }
    }

    #[test]
    fn test_live_sandbox_execution() {
        let tokens = vec![
            CMD_TOOL_SET,
            VAL_BASE,
            VAL_BASE + 7,
            CMD_TOOL_SET,
            VAL_BASE + 1,
            VAL_BASE + 3,
            CMD_TOOL_ADD,
            VAL_BASE,
            VAL_BASE + 1,
            VAL_BASE + 2,
            CMD_TOOL_COPY,
            VAL_BASE + 2,
            VAL_BASE,
            CMD_TOOL_COMMIT,
            EOS_TOKEN,
        ];

        let res = SandboxState::execute_tokens(&tokens);
        assert!(res.is_legal_trace);
        assert!(res.final_state.committed);
        assert_eq!(res.final_state.registers[0], 7);
        assert_eq!(res.final_state.registers[1], 3);
        assert_eq!(res.final_state.registers[2], 10);
        assert_eq!(res.final_state.memory[0], 10);
    }

    #[test]
    fn test_lab_session_stepping() {
        let mut session = LabSession::new("77k", 42, "tool_sandbox", None);
        assert!(!session.prompt_tokens.is_empty());
        assert_eq!(session.generated_tokens.len(), 0);

        session.step(2);
        assert_eq!(session.generated_tokens.len(), 2);
        assert_eq!(session.all_tokens.len(), session.prompt_tokens.len() + 2);
    }

    #[test]
    fn test_command_execution() {
        let mut session = LabSession::new("77k", 42, "tool_sandbox", None);
        assert!(execute_command("step", &mut session));
        assert!(execute_command("regs", &mut session));
        assert!(execute_command("mem", &mut session));
        assert!(execute_command("attn 0 0", &mut session));
        assert!(execute_command("inspect", &mut session));
        assert!(execute_command("tokens", &mut session));
        assert!(execute_command("status", &mut session));
        assert!(execute_command("help", &mut session));
        assert!(execute_command("reset", &mut session));
        assert!(execute_command("task tool_basic", &mut session));
        assert_eq!(session.family_type, TaskFamilyType::ToolWorkflowBasic);
        assert!(!execute_command("quit", &mut session));
    }

    #[test]
    fn test_render_boxes_and_status() {
        let regs = [5, 3, 8, 0];
        let mem = [8, 0, 0, 0];
        let reg_box = render_registers_box(&regs);
        assert!(reg_box.contains("R0"));
        assert!(reg_box.contains("R3"));
        assert!(reg_box.contains("┌"));
        assert!(reg_box.contains("┘"));

        let mem_box = render_memory_box(&mem);
        assert!(mem_box.contains("M0"));
        assert!(mem_box.contains("M3"));

        let mut state = SandboxState::new();
        state.committed = true;
        let status = render_sandbox_status(&state, 5, None);
        assert!(status.contains("COMMITTED"));
        assert!(status.contains("5/32"));

        state.committed = false;
        let err_status = render_sandbox_status(&state, 2, Some("Truncated"));
        assert!(err_status.contains("UNCOMMITTED"));
        assert!(err_status.contains("ERROR"));
        assert!(err_status.contains("Truncated"));
    }

    #[test]
    fn test_attention_renderers() {
        let tokens = vec![BOS_TOKEN, CMD_TOOL_SET, VAL_BASE, SEP_TOKEN];
        let seq_len = tokens.len();
        let weights = vec![0.25; 4 * seq_len * seq_len]; // 4 heads
        let heatmap = render_attention_heatmap(&weights, &tokens, 0, 0, 4);
        assert!(heatmap.contains("Layer 0, Head 0"));
        assert!(heatmap.contains("BOS"));
        assert!(heatmap.contains("SET"));

        let bar_chart = render_attention_bar_chart(&weights, &tokens, 0, 0);
        assert!(bar_chart.contains("Attention Distribution"));
        assert!(bar_chart.contains("SEP"));
    }

    #[test]
    fn test_cli_parser() {
        let cli = LabCli::try_parse_from([
            "lab",
            "--model",
            "1m",
            "--seed",
            "101",
            "--family",
            "fsm_tree",
            "--commands",
            "step;inspect;quit",
            "--batch",
        ])
        .expect("CLI args must parse cleanly");

        assert_eq!(cli.model, "1m");
        assert_eq!(cli.seed, 101);
        assert_eq!(cli.family, "fsm_tree");
        assert_eq!(cli.commands.as_deref(), Some("step;inspect;quit"));
        assert!(cli.batch);
    }
}
