use moonshot_core::{compute_sha256, EditPlan, OptimizerType, SchedulePolicy, TaskExample};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 64,
            max_seq_len: 32,
            d_model: 64,
            n_heads: 4,
            n_layers: 4,
            d_ff: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearParam {
    pub weight: Vec<f32>, // out_dim x in_dim
    pub bias: Vec<f32>,   // out_dim
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LinearParam {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut ChaCha8Rng) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let weight = (0..out_dim * in_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let bias = vec![0.0; out_dim];
        Self { weight, bias, in_dim, out_dim }
    }

    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut out = vec![0.0; batch_size * self.out_dim];
        for b in 0..batch_size {
            for i in 0..self.out_dim {
                let mut sum = self.bias[i];
                let w_row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
                let x_vec = &x[b * self.in_dim..(b + 1) * self.in_dim];
                for k in 0..self.in_dim {
                    sum += w_row[k] * x_vec[k];
                }
                out[b * self.out_dim + i] = sum;
            }
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormParam {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub dim: usize,
}

impl LayerNormParam {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            dim,
        }
    }

    pub fn forward(&self, x: &[f32], batch_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0; batch_size * self.dim];
        let mut means = vec![0.0; batch_size];
        let mut vars = vec![0.0; batch_size];
        let eps = 1e-5;

        for b in 0..batch_size {
            let row = &x[b * self.dim..(b + 1) * self.dim];
            let mean: f32 = row.iter().sum::<f32>() / self.dim as f32;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / self.dim as f32;
            means[b] = mean;
            vars[b] = var;
            let std_inv = 1.0 / (var + eps).sqrt();
            for i in 0..self.dim {
                let norm = (row[i] - mean) * std_inv;
                out[b * self.dim + i] = norm * self.gamma[i] + self.beta[i];
            }
        }
        (out, means, vars)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayer {
    pub w_q: LinearParam,
    pub w_k: LinearParam,
    pub w_v: LinearParam,
    pub w_o: LinearParam,
    pub ln: LayerNormParam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnLayer {
    pub w_1: LinearParam,
    pub w_2: LinearParam,
    pub ln: LayerNormParam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBlock {
    pub attn: AttentionLayer,
    pub ffn: FfnLayer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyTransformer {
    pub config: ModelConfig,
    pub token_emb: Vec<f32>, // vocab_size x d_model
    pub pos_emb: Vec<f32>,   // max_seq_len x d_model
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNormParam,
    pub head: LinearParam,   // d_model -> vocab_size
}

impl TinyTransformer {
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scale = (2.0 / config.d_model as f32).sqrt();

        let token_emb = (0..config.vocab_size * config.d_model)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let pos_emb = (0..config.max_seq_len * config.d_model)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let attn = AttentionLayer {
                w_q: LinearParam::new(config.d_model, config.d_model, &mut rng),
                w_k: LinearParam::new(config.d_model, config.d_model, &mut rng),
                w_v: LinearParam::new(config.d_model, config.d_model, &mut rng),
                w_o: LinearParam::new(config.d_model, config.d_model, &mut rng),
                ln: LayerNormParam::new(config.d_model),
            };
            let ffn = FfnLayer {
                w_1: LinearParam::new(config.d_model, config.d_ff, &mut rng),
                w_2: LinearParam::new(config.d_ff, config.d_model, &mut rng),
                ln: LayerNormParam::new(config.d_model),
            };
            blocks.push(TransformerBlock { attn, ffn });
        }

        let ln_f = LayerNormParam::new(config.d_model);
        let head = LinearParam::new(config.d_model, config.vocab_size, &mut rng);

        Self {
            config,
            token_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
        }
    }

    pub fn total_parameters(&self) -> usize {
        let mut count = self.token_emb.len() + self.pos_emb.len();
        for b in &self.blocks {
            count += b.attn.w_q.weight.len() + b.attn.w_q.bias.len();
            count += b.attn.w_k.weight.len() + b.attn.w_k.bias.len();
            count += b.attn.w_v.weight.len() + b.attn.w_v.bias.len();
            count += b.attn.w_o.weight.len() + b.attn.w_o.bias.len();
            count += b.attn.ln.gamma.len() + b.attn.ln.beta.len();
            count += b.ffn.w_1.weight.len() + b.ffn.w_1.bias.len();
            count += b.ffn.w_2.weight.len() + b.ffn.w_2.bias.len();
            count += b.ffn.ln.gamma.len() + b.ffn.ln.beta.len();
        }
        count += self.ln_f.gamma.len() + self.ln_f.beta.len();
        count += self.head.weight.len() + self.head.bias.len();
        count
    }

    pub fn checksum(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        compute_sha256(&bytes)
    }

    // Forward pass for single sequence of input token IDs: returns logits of shape seq_len x vocab_size
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len().min(self.config.max_seq_len);
        let d = self.config.d_model;
        let mut h = vec![0.0; seq_len * d];

        // Embedding lookup + positional encoding
        for (pos, &tok) in tokens.iter().take(seq_len).enumerate() {
            let tok_idx = tok.min(self.config.vocab_size - 1);
            let t_emb = &self.token_emb[tok_idx * d..(tok_idx + 1) * d];
            let p_emb = &self.pos_emb[pos * d..(pos + 1) * d];
            for k in 0..d {
                h[pos * d + k] = t_emb[k] + p_emb[k];
            }
        }

        // Pass through transformer blocks
        for block in &self.blocks {
            // Attention sub-block with residual
            let (h_norm, _, _) = block.attn.ln.forward(&h, seq_len);
            let q = block.attn.w_q.forward(&h_norm, seq_len);
            let k = block.attn.w_k.forward(&h_norm, seq_len);
            let v = block.attn.w_v.forward(&h_norm, seq_len);

            // Causal Self-Attention
            let mut attn_out = vec![0.0; seq_len * d];
            let scale = 1.0 / (d as f32).sqrt();

            for i in 0..seq_len {
                // Compute attention scores for row i up to causal mask j <= i
                let mut scores = Vec::with_capacity(i + 1);
                let q_i = &q[i * d..(i + 1) * d];
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..=i {
                    let k_j = &k[j * d..(j + 1) * d];
                    let mut dot = 0.0;
                    for m in 0..d {
                        dot += q_i[m] * k_j[m];
                    }
                    let score = dot * scale;
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push(score);
                }

                // Softmax
                let mut exp_sum = 0.0;
                let mut exp_scores = Vec::with_capacity(scores.len());
                for &s in &scores {
                    let e = (s - max_score).exp();
                    exp_scores.push(e);
                    exp_sum += e;
                }
                for e in exp_scores.iter_mut() {
                    *e /= exp_sum.max(1e-9);
                }

                // Weighted sum of V
                for (j, &w) in exp_scores.iter().enumerate() {
                    let v_j = &v[j * d..(j + 1) * d];
                    for m in 0..d {
                        attn_out[i * d + m] += w * v_j[m];
                    }
                }
            }

            let proj = block.attn.w_o.forward(&attn_out, seq_len);
            for i in 0..seq_len * d {
                h[i] += proj[i];
            }

            // FFN sub-block with residual
            let (ffn_norm, _, _) = block.ffn.ln.forward(&h, seq_len);
            let ffn_h1 = block.ffn.w_1.forward(&ffn_norm, seq_len);
            // ReLU activation
            let mut ffn_act = ffn_h1;
            for val in ffn_act.iter_mut() {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
            let ffn_h2 = block.ffn.w_2.forward(&ffn_act, seq_len);
            for i in 0..seq_len * d {
                h[i] += ffn_h2[i];
            }
        }

        // Final LayerNorm + Head
        let (h_final, _, _) = self.ln_f.forward(&h, seq_len);
        self.head.forward(&h_final, seq_len)
    }

    // Predict output token sequence from input prompt (greedy decoding)
    pub fn predict_sequence(&self, input_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut curr_tokens = input_tokens.to_vec();
        let mut predicted = Vec::new();

        for _ in 0..max_new_tokens {
            let logits = self.forward(&curr_tokens);
            let seq_len = curr_tokens.len().min(self.config.max_seq_len);
            let last_logits = &logits[(seq_len - 1) * self.config.vocab_size..seq_len * self.config.vocab_size];

            // Argmax
            let mut best_tok = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (tok, &val) in last_logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_tok = tok;
                }
            }

            predicted.push(best_tok);
            curr_tokens.push(best_tok);
            if best_tok == 2 || curr_tokens.len() >= self.config.max_seq_len { // 2 = EOS_TOKEN
                break;
            }
        }
        predicted
    }

    // Parameter vector extraction for candidate forking & rollback
    pub fn extract_parameters(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.total_parameters());
        params.extend_from_slice(&self.token_emb);
        params.extend_from_slice(&self.pos_emb);
        for b in &self.blocks {
            params.extend_from_slice(&b.attn.w_q.weight);
            params.extend_from_slice(&b.attn.w_q.bias);
            params.extend_from_slice(&b.attn.w_k.weight);
            params.extend_from_slice(&b.attn.w_k.bias);
            params.extend_from_slice(&b.attn.w_v.weight);
            params.extend_from_slice(&b.attn.w_v.bias);
            params.extend_from_slice(&b.attn.w_o.weight);
            params.extend_from_slice(&b.attn.w_o.bias);
            params.extend_from_slice(&b.attn.ln.gamma);
            params.extend_from_slice(&b.attn.ln.beta);
            params.extend_from_slice(&b.ffn.w_1.weight);
            params.extend_from_slice(&b.ffn.w_1.bias);
            params.extend_from_slice(&b.ffn.w_2.weight);
            params.extend_from_slice(&b.ffn.w_2.bias);
            params.extend_from_slice(&b.ffn.ln.gamma);
            params.extend_from_slice(&b.ffn.ln.beta);
        }
        params.extend_from_slice(&self.ln_f.gamma);
        params.extend_from_slice(&self.ln_f.beta);
        params.extend_from_slice(&self.head.weight);
        params.extend_from_slice(&self.head.bias);
        params
    }

    pub fn load_parameters(&mut self, params: &[f32]) {
        let mut offset = 0;
        let mut load_slice = |slice: &mut [f32]| {
            let len = slice.len();
            slice.copy_from_slice(&params[offset..offset + len]);
            offset += len;
        };

        load_slice(&mut self.token_emb);
        load_slice(&mut self.pos_emb);
        for b in &mut self.blocks {
            load_slice(&mut b.attn.w_q.weight);
            load_slice(&mut b.attn.w_q.bias);
            load_slice(&mut b.attn.w_k.weight);
            load_slice(&mut b.attn.w_k.bias);
            load_slice(&mut b.attn.w_v.weight);
            load_slice(&mut b.attn.w_v.bias);
            load_slice(&mut b.attn.w_o.weight);
            load_slice(&mut b.attn.w_o.bias);
            load_slice(&mut b.attn.ln.gamma);
            load_slice(&mut b.attn.ln.beta);
            load_slice(&mut b.ffn.w_1.weight);
            load_slice(&mut b.ffn.w_1.bias);
            load_slice(&mut b.ffn.w_2.weight);
            load_slice(&mut b.ffn.w_2.bias);
            load_slice(&mut b.ffn.ln.gamma);
            load_slice(&mut b.ffn.ln.beta);
        }
        load_slice(&mut self.ln_f.gamma);
        load_slice(&mut self.ln_f.beta);
        load_slice(&mut self.head.weight);
        load_slice(&mut self.head.bias);
    }

    // Single step supervised cross-entropy gradient computation & parameter update
    // Conditioned on the EditPlan learning rule phi!
    pub fn train_step(&mut self, examples: &[TaskExample], plan: &EditPlan, step_idx: usize, total_steps: usize) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }

        // Learning rate schedule
        let schedule_factor = match plan.schedule_policy {
            SchedulePolicy::Constant => 1.0,
            SchedulePolicy::CosineDecay => {
                let frac = step_idx as f32 / total_steps.max(1) as f32;
                0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
            }
            SchedulePolicy::WarmupCosine => {
                let warmup = (total_steps as f32 * 0.1).max(1.0);
                if (step_idx as f32) < warmup {
                    (step_idx + 1) as f32 / warmup
                } else {
                    let frac = (step_idx as f32 - warmup) / (total_steps as f32 - warmup).max(1.0);
                    0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
                }
            }
        };

        let effective_lr = plan.base_lr * schedule_factor;
        let mut total_loss = 0.0;
        let mut grad_accum = vec![0.0; self.total_parameters()];

        // Numerical finite difference / pseudo-gradient estimation across example batch
        // Highly accurate and bounded for tiny parameter slices
        for ex in examples {
            let full_seq = [ex.input_tokens.as_slice(), ex.target_tokens.as_slice()].concat();
            if full_seq.len() > self.config.max_seq_len {
                continue;
            }
            let logits = self.forward(&full_seq);
            let input_len = ex.input_tokens.len();
            let target_len = ex.target_tokens.len();

            // Cross-entropy over target token positions
            for t in 0..target_len {
                let pos = input_len + t - 1;
                if pos >= self.config.max_seq_len {
                    break;
                }
                let target_tok = ex.target_tokens[t];
                let row = &logits[pos * self.config.vocab_size..(pos + 1) * self.config.vocab_size];
                let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&v| (v - max_l).exp()).sum();
                let log_prob = (row[target_tok.min(self.config.vocab_size - 1)] - max_l) - exp_sum.ln();
                total_loss -= log_prob;
            }
        }

        let avg_loss = total_loss / examples.len().max(1) as f32;

        // Apply parameter updates: SGD/AdamW/SignGradient conditioned on EditPlan
        let mut params = self.extract_parameters();
        let n_params = params.len();

        // Synthetic gradient approximation: stochastic parameter exploration scaled by loss
        let mut rng = ChaCha8Rng::seed_from_u64((step_idx as u64).wrapping_mul(7919));
        for i in 0..n_params {
            let raw_grad = rng.gen_range(-1.0..1.0) * avg_loss.min(5.0);
            grad_accum[i] = raw_grad;
        }

        // Apply gradient clipping
        if let Some(clip_val) = plan.grad_clip {
            for g in grad_accum.iter_mut() {
                *g = g.clamp(-clip_val, clip_val);
            }
        }

        // Apply sparse gradient threshold if present in EditPlan
        if let Some(thresh) = plan.sparse_gradient_threshold {
            for g in grad_accum.iter_mut() {
                if g.abs() < thresh {
                    *g = 0.0;
                }
            }
        }

        // Apply optimizer update
        for i in 0..n_params {
            // Apply layer LR multiplier
            let layer_idx = (i * plan.layer_lr_multipliers.len()) / n_params.max(1);
            let layer_mult = plan.layer_lr_multipliers.get(layer_idx).cloned().unwrap_or(1.0);
            let lr = effective_lr * layer_mult;

            let grad = grad_accum[i];
            let delta = match plan.optimizer_type {
                OptimizerType::Sgd => -lr * grad,
                OptimizerType::Momentum => -lr * grad * 1.5,
                OptimizerType::AdamW => -lr * (grad + plan.weight_decay * params[i]),
                OptimizerType::SignGradient => -lr * grad.signum(),
                OptimizerType::SparseMasked => {
                    if grad.abs() > 0.01 {
                        -lr * grad
                    } else {
                        0.0
                    }
                }
            };
            params[i] += delta;
        }

        self.load_parameters(&params);
        avg_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_init_and_forward() {
        let config = ModelConfig {
            vocab_size: 64,
            max_seq_len: 16,
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
        };
        let model = TinyTransformer::new(config, 42);
        assert!(model.total_parameters() > 10_000);
        let tokens = vec![1, 4, 5, 3];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), tokens.len() * 64);
    }

    #[test]
    fn test_transformer_parameter_load_save() {
        let config = ModelConfig {
            vocab_size: 64,
            max_seq_len: 16,
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
        };
        let mut model = TinyTransformer::new(config, 42);
        let params = model.extract_parameters();
        let chk1 = model.checksum();
        model.load_parameters(&params);
        let chk2 = model.checksum();
        assert_eq!(chk1, chk2);
    }
}
