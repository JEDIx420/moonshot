use moonshot_core::{compute_sha256, EditPlan, OptimizerType, SamplingPolicy, SchedulePolicy, TaskExample};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
            n_layers: 2,
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

    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = vec![0.0; seq_len * self.out_dim];
        for t in 0..seq_len {
            for i in 0..self.out_dim {
                let mut sum = self.bias[i];
                let w_row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
                let x_vec = &x[t * self.in_dim..(t + 1) * self.in_dim];
                for k in 0..self.in_dim {
                    sum += w_row[k] * x_vec[k];
                }
                out[t * self.out_dim + i] = sum;
            }
        }
        out
    }

    pub fn backward(
        &self,
        dy: &[f32],
        x: &[f32],
        seq_len: usize,
        dx: &mut [f32],
        dw: &mut [f32],
        db: &mut [f32],
    ) {
        for t in 0..seq_len {
            let dy_t = &dy[t * self.out_dim..(t + 1) * self.out_dim];
            let x_t = &x[t * self.in_dim..(t + 1) * self.in_dim];

            for i in 0..self.out_dim {
                db[i] += dy_t[i];
                let w_row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
                for k in 0..self.in_dim {
                    dw[i * self.in_dim + k] += dy_t[i] * x_t[k];
                    dx[t * self.in_dim + k] += dy_t[i] * w_row[k];
                }
            }
        }
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

    pub fn forward(&self, x: &[f32], seq_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0; seq_len * self.dim];
        let mut means = vec![0.0; seq_len];
        let mut vars = vec![0.0; seq_len];
        let eps = 1e-5;

        for t in 0..seq_len {
            let row = &x[t * self.dim..(t + 1) * self.dim];
            let mean: f32 = row.iter().sum::<f32>() / self.dim as f32;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / self.dim as f32;
            means[t] = mean;
            vars[t] = var;
            let std_inv = 1.0 / (var + eps).sqrt();
            for i in 0..self.dim {
                let norm = (row[i] - mean) * std_inv;
                out[t * self.dim + i] = norm * self.gamma[i] + self.beta[i];
            }
        }
        (out, means, vars)
    }

    pub fn backward(
        &self,
        dy: &[f32],
        x: &[f32],
        means: &[f32],
        vars: &[f32],
        seq_len: usize,
        dx: &mut [f32],
        dgamma: &mut [f32],
        dbeta: &mut [f32],
    ) {
        let eps = 1e-5;
        let d = self.dim as f32;

        for t in 0..seq_len {
            let mean = means[t];
            let var = vars[t];
            let std_inv = 1.0 / (var + eps).sqrt();
            let row_x = &x[t * self.dim..(t + 1) * self.dim];
            let row_dy = &dy[t * self.dim..(t + 1) * self.dim];

            let mut dnorm = vec![0.0; self.dim];
            for i in 0..self.dim {
                dgamma[i] += row_dy[i] * (row_x[i] - mean) * std_inv;
                dbeta[i] += row_dy[i];
                dnorm[i] = row_dy[i] * self.gamma[i];
            }

            let sum_dnorm: f32 = dnorm.iter().sum();
            let sum_dnorm_norm: f32 = dnorm.iter().zip(row_x.iter()).map(|(&dn, &x_i)| dn * (x_i - mean) * std_inv).sum();

            for i in 0..self.dim {
                let norm_i = (row_x[i] - mean) * std_inv;
                dx[t * self.dim + i] += std_inv * (dnorm[i] - (sum_dnorm + norm_i * sum_dnorm_norm) / d);
            }
        }
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
pub struct ModelCheckpoint {
    pub config: ModelConfig,
    pub parameters: Vec<f32>,
    pub adam_m: Vec<f32>,
    pub adam_v: Vec<f32>,
    pub opt_step: usize,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyTransformer {
    pub config: ModelConfig,
    pub token_emb: Vec<f32>, // vocab_size x d_model
    pub pos_emb: Vec<f32>,   // max_seq_len x d_model
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNormParam,
    pub head: LinearParam,   // d_model -> vocab_size
    // Optimizer moments
    #[serde(skip)]
    pub adam_m: Vec<f32>,
    #[serde(skip)]
    pub adam_v: Vec<f32>,
    #[serde(skip)]
    pub opt_step: usize,
}

// Multi-head causal attention helper routines
pub fn forward_multihead_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d_k = d_model / n_heads;
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut attn_out = vec![0.0; seq_len * d_model];
    let mut attn_weights = vec![0.0; n_heads * seq_len * seq_len];

    for h in 0..n_heads {
        for i in 0..seq_len {
            let q_offset = i * d_model + h * d_k;
            let q_h_i = &q[q_offset..q_offset + d_k];

            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(i + 1);

            for j in 0..=i {
                let k_offset = j * d_model + h * d_k;
                let k_h_j = &k[k_offset..k_offset + d_k];
                let mut dot = 0.0;
                for m in 0..d_k {
                    dot += q_h_i[m] * k_h_j[m];
                }
                let s = dot * scale;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }

            let mut exp_sum = 0.0;
            let mut exp_scores = Vec::with_capacity(i + 1);
            for &s in &scores {
                let e = (s - max_score).exp();
                exp_scores.push(e);
                exp_sum += e;
            }

            let denom = exp_sum.max(1e-9);
            for j in 0..=i {
                let w = exp_scores[j] / denom;
                attn_weights[h * seq_len * seq_len + i * seq_len + j] = w;

                let v_offset = j * d_model + h * d_k;
                let v_h_j = &v[v_offset..v_offset + d_k];
                let out_offset = i * d_model + h * d_k;
                for m in 0..d_k {
                    attn_out[out_offset + m] += w * v_h_j[m];
                }
            }
        }
    }
    (attn_out, attn_weights)
}

pub fn backward_multihead_attention(
    d_attn_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_weights: &[f32],
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
) {
    let d_k_dim = d_model / n_heads;
    let scale = 1.0 / (d_k_dim as f32).sqrt();

    for h in 0..n_heads {
        for i in 0..seq_len {
            let out_offset = i * d_model + h * d_k_dim;
            let d_out_h_i = &d_attn_out[out_offset..out_offset + d_k_dim];

            let mut d_a = vec![0.0; i + 1];
            let mut a_times_da_sum = 0.0;

            for j in 0..=i {
                let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
                let v_offset = j * d_model + h * d_k_dim;
                let v_h_j = &v[v_offset..v_offset + d_k_dim];

                let mut dot_da = 0.0;
                for m in 0..d_k_dim {
                    dot_da += d_out_h_i[m] * v_h_j[m];
                    d_v[v_offset + m] += w * d_out_h_i[m];
                }
                d_a[j] = dot_da;
                a_times_da_sum += w * dot_da;
            }

            let q_offset = i * d_model + h * d_k_dim;
            for j in 0..=i {
                let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
                let d_s_ij = w * (d_a[j] - a_times_da_sum);

                let k_offset = j * d_model + h * d_k_dim;
                for m in 0..d_k_dim {
                    d_q[q_offset + m] += scale * d_s_ij * k[k_offset + m];
                    d_k[k_offset + m] += scale * d_s_ij * q[q_offset + m];
                }
            }
        }
    }
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

        let mut model = Self {
            config,
            token_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            opt_step: 0,
        };
        let total = model.total_parameters();
        model.adam_m = vec![0.0; total];
        model.adam_v = vec![0.0; total];
        model
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

    pub fn save_checkpoint(&self) -> ModelCheckpoint {
        let parameters = self.extract_parameters();
        let checksum = self.checksum();
        ModelCheckpoint {
            config: self.config.clone(),
            parameters,
            adam_m: self.adam_m.clone(),
            adam_v: self.adam_v.clone(),
            opt_step: self.opt_step,
            checksum,
        }
    }

    pub fn load_checkpoint(&mut self, ckpt: &ModelCheckpoint) -> Result<(), String> {
        if ckpt.parameters.len() != self.total_parameters() {
            return Err(format!(
                "Checkpoint parameter size mismatch: expected {}, got {}",
                self.total_parameters(),
                ckpt.parameters.len()
            ));
        }
        self.load_parameters(&ckpt.parameters);
        self.adam_m = ckpt.adam_m.clone();
        self.adam_v = ckpt.adam_v.clone();
        self.opt_step = ckpt.opt_step;
        Ok(())
    }

    pub fn parameter_groups(&self) -> Vec<usize> {
        let mut groups = Vec::with_capacity(self.total_parameters());
        // Group 0: token_emb + pos_emb
        let emb_count = self.token_emb.len() + self.pos_emb.len();
        groups.resize(emb_count, 0);

        for (b_idx, b) in self.blocks.iter().enumerate() {
            let attn_group = 1 + b_idx * 2;
            let ffn_group = 2 + b_idx * 2;

            let attn_count = b.attn.w_q.weight.len() + b.attn.w_q.bias.len()
                + b.attn.w_k.weight.len() + b.attn.w_k.bias.len()
                + b.attn.w_v.weight.len() + b.attn.w_v.bias.len()
                + b.attn.w_o.weight.len() + b.attn.w_o.bias.len()
                + b.attn.ln.gamma.len() + b.attn.ln.beta.len();
            groups.resize(groups.len() + attn_count, attn_group);

            let ffn_count = b.ffn.w_1.weight.len() + b.ffn.w_1.bias.len()
                + b.ffn.w_2.weight.len() + b.ffn.w_2.bias.len()
                + b.ffn.ln.gamma.len() + b.ffn.ln.beta.len();
            groups.resize(groups.len() + ffn_count, ffn_group);
        }

        // Final group 5: ln_f + head
        let final_count = self.ln_f.gamma.len() + self.ln_f.beta.len() + self.head.weight.len() + self.head.bias.len();
        groups.resize(groups.len() + final_count, 5);
        groups
    }

    // Forward pass
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len().min(self.config.max_seq_len);
        let d = self.config.d_model;
        let mut h = vec![0.0; seq_len * d];

        for (pos, &tok) in tokens.iter().take(seq_len).enumerate() {
            let tok_idx = tok.min(self.config.vocab_size - 1);
            let t_emb = &self.token_emb[tok_idx * d..(tok_idx + 1) * d];
            let p_emb = &self.pos_emb[pos * d..(pos + 1) * d];
            for k in 0..d {
                h[pos * d + k] = t_emb[k] + p_emb[k];
            }
        }

        for block in &self.blocks {
            // Multi-head causal attention sub-block with residual
            let (h_norm, _, _) = block.attn.ln.forward(&h, seq_len);
            let q = block.attn.w_q.forward(&h_norm, seq_len);
            let k = block.attn.w_k.forward(&h_norm, seq_len);
            let v = block.attn.w_v.forward(&h_norm, seq_len);

            let (attn_out, _) = forward_multihead_attention(
                &q, &k, &v, seq_len, d, self.config.n_heads
            );

            let proj = block.attn.w_o.forward(&attn_out, seq_len);
            for i in 0..seq_len * d {
                h[i] += proj[i];
            }

            // FFN sub-block with residual
            let (ffn_norm, _, _) = block.ffn.ln.forward(&h, seq_len);
            let ffn_h1 = block.ffn.w_1.forward(&ffn_norm, seq_len);
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

        let (h_final, _, _) = self.ln_f.forward(&h, seq_len);
        self.head.forward(&h_final, seq_len)
    }

    pub fn predict_sequence(&self, input_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut curr_tokens = input_tokens.to_vec();
        let mut predicted = Vec::new();

        for _ in 0..max_new_tokens {
            let logits = self.forward(&curr_tokens);
            let seq_len = curr_tokens.len().min(self.config.max_seq_len);
            let last_logits = &logits[(seq_len - 1) * self.config.vocab_size..seq_len * self.config.vocab_size];

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
            if best_tok == 2 || curr_tokens.len() >= self.config.max_seq_len {
                break;
            }
        }
        predicted
    }

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

    // Exact analytical gradient and loss computation across examples
    pub fn compute_loss_and_gradients(&self, examples: &[TaskExample]) -> (f32, Vec<f32>) {
        let d = self.config.d_model;
        let v = self.config.vocab_size;

        let mut d_token_emb = vec![0.0; self.token_emb.len()];
        let mut d_pos_emb = vec![0.0; self.pos_emb.len()];
        let mut d_ln_f_gamma = vec![0.0; self.ln_f.gamma.len()];
        let mut d_ln_f_beta = vec![0.0; self.ln_f.beta.len()];
        let mut d_head_w = vec![0.0; self.head.weight.len()];
        let mut d_head_b = vec![0.0; self.head.bias.len()];

        struct BlockGrads {
            d_wq_w: Vec<f32>, d_wq_b: Vec<f32>,
            d_wk_w: Vec<f32>, d_wk_b: Vec<f32>,
            d_wv_w: Vec<f32>, d_wv_b: Vec<f32>,
            d_wo_w: Vec<f32>, d_wo_b: Vec<f32>,
            d_attn_ln_gamma: Vec<f32>, d_attn_ln_beta: Vec<f32>,
            d_w1_w: Vec<f32>, d_w1_b: Vec<f32>,
            d_w2_w: Vec<f32>, d_w2_b: Vec<f32>,
            d_ffn_ln_gamma: Vec<f32>, d_ffn_ln_beta: Vec<f32>,
        }

        let mut block_grads: Vec<BlockGrads> = self.blocks.iter().map(|b| BlockGrads {
            d_wq_w: vec![0.0; b.attn.w_q.weight.len()], d_wq_b: vec![0.0; b.attn.w_q.bias.len()],
            d_wk_w: vec![0.0; b.attn.w_k.weight.len()], d_wk_b: vec![0.0; b.attn.w_k.bias.len()],
            d_wv_w: vec![0.0; b.attn.w_v.weight.len()], d_wv_b: vec![0.0; b.attn.w_v.bias.len()],
            d_wo_w: vec![0.0; b.attn.w_o.weight.len()], d_wo_b: vec![0.0; b.attn.w_o.bias.len()],
            d_attn_ln_gamma: vec![0.0; b.attn.ln.gamma.len()], d_attn_ln_beta: vec![0.0; b.attn.ln.beta.len()],
            d_w1_w: vec![0.0; b.ffn.w_1.weight.len()], d_w1_b: vec![0.0; b.ffn.w_1.bias.len()],
            d_w2_w: vec![0.0; b.ffn.w_2.weight.len()], d_w2_b: vec![0.0; b.ffn.w_2.bias.len()],
            d_ffn_ln_gamma: vec![0.0; b.ffn.ln.gamma.len()], d_ffn_ln_beta: vec![0.0; b.ffn.ln.beta.len()],
        }).collect();

        let mut total_loss = 0.0;
        let mut total_tokens_scored = 0;

        for ex in examples {
            let full_seq = [ex.input_tokens.as_slice(), ex.target_tokens.as_slice()].concat();
            let seq_len = full_seq.len().min(self.config.max_seq_len);
            if seq_len < 2 {
                continue;
            }

            let mut h = vec![0.0; seq_len * d];
            for (pos, &tok) in full_seq.iter().take(seq_len).enumerate() {
                let tok_idx = tok.min(v - 1);
                let t_emb = &self.token_emb[tok_idx * d..(tok_idx + 1) * d];
                let p_emb = &self.pos_emb[pos * d..(pos + 1) * d];
                for k in 0..d {
                    h[pos * d + k] = t_emb[k] + p_emb[k];
                }
            }

            struct BlockCache {
                h_in: Vec<f32>,
                attn_ln_out: Vec<f32>,
                attn_means: Vec<f32>,
                attn_vars: Vec<f32>,
                q: Vec<f32>,
                k: Vec<f32>,
                v: Vec<f32>,
                attn_weights: Vec<f32>,
                attn_out: Vec<f32>,
                h_mid: Vec<f32>,
                ffn_ln_out: Vec<f32>,
                ffn_means: Vec<f32>,
                ffn_vars: Vec<f32>,
                ffn_h1: Vec<f32>,
                ffn_act: Vec<f32>,
            }

            let mut block_caches = Vec::with_capacity(self.blocks.len());

            for block in &self.blocks {
                let h_in = h.clone();
                let (attn_ln_out, attn_means, attn_vars) = block.attn.ln.forward(&h, seq_len);
                let q = block.attn.w_q.forward(&attn_ln_out, seq_len);
                let k = block.attn.w_k.forward(&attn_ln_out, seq_len);
                let v_val = block.attn.w_v.forward(&attn_ln_out, seq_len);

                let (attn_out, attn_weights) = forward_multihead_attention(
                    &q, &k, &v_val, seq_len, d, self.config.n_heads
                );

                let proj = block.attn.w_o.forward(&attn_out, seq_len);
                for i in 0..seq_len * d {
                    h[i] += proj[i];
                }
                let h_mid = h.clone();

                let (ffn_ln_out, ffn_means, ffn_vars) = block.ffn.ln.forward(&h, seq_len);
                let ffn_h1 = block.ffn.w_1.forward(&ffn_ln_out, seq_len);
                let mut ffn_act = ffn_h1.clone();
                for val in ffn_act.iter_mut() {
                    if *val < 0.0 { *val = 0.0; }
                }
                let ffn_h2 = block.ffn.w_2.forward(&ffn_act, seq_len);
                for i in 0..seq_len * d {
                    h[i] += ffn_h2[i];
                }

                block_caches.push(BlockCache {
                    h_in,
                    attn_ln_out,
                    attn_means,
                    attn_vars,
                    q,
                    k,
                    v: v_val,
                    attn_weights,
                    attn_out,
                    h_mid,
                    ffn_ln_out,
                    ffn_means,
                    ffn_vars,
                    ffn_h1,
                    ffn_act,
                });
            }

            let (h_final, ln_f_means, ln_f_vars) = self.ln_f.forward(&h, seq_len);
            let logits = self.head.forward(&h_final, seq_len);

            let mut d_logits = vec![0.0; seq_len * v];
            let input_len = ex.input_tokens.len();

            for t in 0..ex.target_tokens.len() {
                let pos = input_len + t - 1;
                if pos >= seq_len { break; }
                let target_tok = ex.target_tokens[t].min(v - 1);
                let row = &logits[pos * v..(pos + 1) * v];
                let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&val| (val - max_l).exp()).sum();

                for c in 0..v {
                    let prob = (row[c] - max_l).exp() / exp_sum.max(1e-9);
                    let target_ind = if c == target_tok { 1.0 } else { 0.0 };
                    d_logits[pos * v + c] = prob - target_ind;
                }

                let target_prob = (row[target_tok] - max_l).exp() / exp_sum.max(1e-9);
                total_loss -= target_prob.max(1e-9).ln();
                total_tokens_scored += 1;
            }

            // Backward through Head
            let mut d_h_final = vec![0.0; seq_len * d];
            self.head.backward(&d_logits, &h_final, seq_len, &mut d_h_final, &mut d_head_w, &mut d_head_b);

            // Backward through Final LayerNorm
            let mut d_h = vec![0.0; seq_len * d];
            self.ln_f.backward(&d_h_final, &h, &ln_f_means, &ln_f_vars, seq_len, &mut d_h, &mut d_ln_f_gamma, &mut d_ln_f_beta);

            // Backward through Transformer blocks
            for (b_idx, block) in self.blocks.iter().enumerate().rev() {
                let cache = &block_caches[b_idx];
                let bg = &mut block_grads[b_idx];

                // FFN backward
                let d_ffn_h2 = d_h.clone();
                let mut d_ffn_act = vec![0.0; seq_len * self.config.d_ff];
                block.ffn.w_2.backward(&d_ffn_h2, &cache.ffn_act, seq_len, &mut d_ffn_act, &mut bg.d_w2_w, &mut bg.d_w2_b);

                let mut d_ffn_h1 = d_ffn_act;
                for i in 0..d_ffn_h1.len() {
                    if cache.ffn_h1[i] <= 0.0 { d_ffn_h1[i] = 0.0; }
                }

                let mut d_ffn_ln_out = vec![0.0; seq_len * d];
                block.ffn.w_1.backward(&d_ffn_h1, &cache.ffn_ln_out, seq_len, &mut d_ffn_ln_out, &mut bg.d_w1_w, &mut bg.d_w1_b);

                let mut d_h_mid_branch = vec![0.0; seq_len * d];
                block.ffn.ln.backward(&d_ffn_ln_out, &cache.h_mid, &cache.ffn_means, &cache.ffn_vars, seq_len, &mut d_h_mid_branch, &mut bg.d_ffn_ln_gamma, &mut bg.d_ffn_ln_beta);

                for i in 0..seq_len * d {
                    d_h[i] += d_h_mid_branch[i];
                }

                // Attention backward
                let d_attn_proj = d_h.clone();
                let mut d_attn_out = vec![0.0; seq_len * d];
                block.attn.w_o.backward(&d_attn_proj, &cache.attn_out, seq_len, &mut d_attn_out, &mut bg.d_wo_w, &mut bg.d_wo_b);

                let mut d_q = vec![0.0; seq_len * d];
                let mut d_k = vec![0.0; seq_len * d];
                let mut d_v = vec![0.0; seq_len * d];

                backward_multihead_attention(
                    &d_attn_out,
                    &cache.q,
                    &cache.k,
                    &cache.v,
                    &cache.attn_weights,
                    seq_len,
                    d,
                    self.config.n_heads,
                    &mut d_q,
                    &mut d_k,
                    &mut d_v,
                );

                let mut d_attn_ln_out = vec![0.0; seq_len * d];
                block.attn.w_q.backward(&d_q, &cache.attn_ln_out, seq_len, &mut d_attn_ln_out, &mut bg.d_wq_w, &mut bg.d_wq_b);
                block.attn.w_k.backward(&d_k, &cache.attn_ln_out, seq_len, &mut d_attn_ln_out, &mut bg.d_wk_w, &mut bg.d_wk_b);
                block.attn.w_v.backward(&d_v, &cache.attn_ln_out, seq_len, &mut d_attn_ln_out, &mut bg.d_wv_w, &mut bg.d_wv_b);

                let mut d_h_in_branch = vec![0.0; seq_len * d];
                block.attn.ln.backward(&d_attn_ln_out, &cache.h_in, &cache.attn_means, &cache.attn_vars, seq_len, &mut d_h_in_branch, &mut bg.d_attn_ln_gamma, &mut bg.d_attn_ln_beta);

                for i in 0..seq_len * d {
                    d_h[i] += d_h_in_branch[i];
                }
            }

            // Embedding backward
            for (pos, &tok) in full_seq.iter().take(seq_len).enumerate() {
                let tok_idx = tok.min(v - 1);
                for m in 0..d {
                    let grad = d_h[pos * d + m];
                    d_token_emb[tok_idx * d + m] += grad;
                    d_pos_emb[pos * d + m] += grad;
                }
            }
        }

        let norm_factor = total_tokens_scored.max(1) as f32;
        let mut all_grads = Vec::with_capacity(self.total_parameters());
        all_grads.extend(d_token_emb.iter().map(|g| g / norm_factor));
        all_grads.extend(d_pos_emb.iter().map(|g| g / norm_factor));

        for bg in &block_grads {
            all_grads.extend(bg.d_wq_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wq_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wk_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wk_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wv_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wv_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wo_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_wo_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_attn_ln_gamma.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_attn_ln_beta.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_w1_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_w1_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_w2_w.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_w2_b.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_ffn_ln_gamma.iter().map(|g| g / norm_factor));
            all_grads.extend(bg.d_ffn_ln_beta.iter().map(|g| g / norm_factor));
        }
        all_grads.extend(d_ln_f_gamma.iter().map(|g| g / norm_factor));
        all_grads.extend(d_ln_f_beta.iter().map(|g| g / norm_factor));
        all_grads.extend(d_head_w.iter().map(|g| g / norm_factor));
        all_grads.extend(d_head_b.iter().map(|g| g / norm_factor));

        let avg_loss = if total_tokens_scored > 0 {
            total_loss / total_tokens_scored as f32
        } else {
            0.0
        };

        (avg_loss, all_grads)
    }

    // Exact analytical backpropagation and parameter update conditioned on EditPlan
    pub fn train_step(&mut self, examples: &[TaskExample], plan: &EditPlan, step_idx: usize, total_steps: usize) -> f32 {
        self.train_step_projected(examples, None, plan, step_idx, total_steps)
    }

    /// Wave 3: Parameter update with Orthogonal Gradient Projection (A-GEM) to prevent catastrophic forgetting
    pub fn train_step_projected(
        &mut self,
        examples: &[TaskExample],
        anchor_examples: Option<&[TaskExample]>,
        plan: &EditPlan,
        step_idx: usize,
        total_steps: usize,
    ) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }

        // Sampling policy selection
        let selected_examples: Vec<TaskExample> = match plan.sampling_policy {
            SamplingPolicy::Uniform => examples.to_vec(),
            SamplingPolicy::HardMining | SamplingPolicy::ErrorWeighted => {
                let mut filtered = Vec::new();
                for ex in examples {
                    let pred = self.predict_sequence(&ex.input_tokens, ex.target_tokens.len());
                    if pred != ex.target_tokens {
                        filtered.push(ex.clone());
                    }
                }
                if filtered.is_empty() {
                    examples.to_vec()
                } else {
                    filtered
                }
            }
        };

        // Learning rate schedule
        let schedule_factor = match plan.schedule_policy {
            SchedulePolicy::Constant => 1.0,
            SchedulePolicy::CosineDecay => {
                let frac = step_idx as f32 / total_steps.max(1) as f32;
                0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
            }
            SchedulePolicy::WarmupCosine => {
                let warmup = (total_steps as f32 * 0.15).max(1.0);
                if (step_idx as f32) < warmup {
                    (step_idx + 1) as f32 / warmup
                } else {
                    let frac = (step_idx as f32 - warmup) / (total_steps as f32 - warmup).max(1.0);
                    0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
                }
            }
        };

        let effective_lr = plan.base_lr * schedule_factor;
        let (avg_loss, mut all_grads) = self.compute_loss_and_gradients(&selected_examples);

        // Orthogonal gradient projection against anchor set (A-GEM)
        if let Some(anchors) = anchor_examples {
            if !anchors.is_empty() {
                let (_, anchor_grads) = self.compute_loss_and_gradients(anchors);
                let dot: f32 = all_grads.iter().zip(anchor_grads.iter()).map(|(&g1, &g2)| g1 * g2).sum();
                if dot < 0.0 {
                    let anchor_norm_sq: f32 = anchor_grads.iter().map(|&g| g * g).sum();
                    if anchor_norm_sq > 1e-8 {
                        let proj_coeff = dot / anchor_norm_sq;
                        for i in 0..all_grads.len() {
                            all_grads[i] -= proj_coeff * anchor_grads[i];
                        }
                    }
                }
            }
        }

        // Global norm gradient clipping
        if let Some(clip) = plan.grad_clip {
            let norm_sq: f32 = all_grads.iter().map(|&g| g * g).sum();
            let norm = norm_sq.sqrt();
            if norm > clip && norm > 1e-7 {
                let scale = clip / norm;
                for g in all_grads.iter_mut() {
                    *g *= scale;
                }
            }
        }

        // Sparse gradient mask thresholding
        if let Some(thresh) = plan.sparse_gradient_threshold {
            for g in all_grads.iter_mut() {
                if g.abs() < thresh { *g = 0.0; }
            }
        }

        // Apply optimizer update
        let mut params = self.extract_parameters();
        let n_params = params.len();
        let param_groups = self.parameter_groups();
        self.opt_step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        for i in 0..n_params {
            let group = param_groups[i];
            let layer_mult = plan.layer_lr_multipliers.get(group).cloned().unwrap_or(1.0);
            let lr = effective_lr * layer_mult;
            let g = all_grads[i];

            let delta = match plan.optimizer_type {
                OptimizerType::Sgd => -lr * (g + plan.weight_decay * params[i]),
                OptimizerType::Momentum => {
                    self.adam_m[i] = beta1 * self.adam_m[i] + g;
                    -lr * (self.adam_m[i] + plan.weight_decay * params[i])
                }
                OptimizerType::AdamW => {
                    self.adam_m[i] = beta1 * self.adam_m[i] + (1.0 - beta1) * g;
                    self.adam_v[i] = beta2 * self.adam_v[i] + (1.0 - beta2) * g * g;
                    let m_hat = self.adam_m[i] / (1.0 - beta1.powi(self.opt_step as i32)).max(1e-6);
                    let v_hat = self.adam_v[i] / (1.0 - beta2.powi(self.opt_step as i32)).max(1e-6);
                    -lr * (m_hat / (v_hat.sqrt() + eps) + plan.weight_decay * params[i])
                }
                OptimizerType::SignGradient => -lr * g.signum(),
                OptimizerType::SparseMasked => {
                    if g.abs() > 0.005 { -lr * g } else { 0.0 }
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
        let config = ModelConfig::default();
        let model = TinyTransformer::new(config, 42);
        assert_eq!(model.total_parameters(), 77376);
        let tokens = vec![1, 4, 5, 3];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), tokens.len() * 64);
    }

    #[test]
    fn test_checkpoint_roundtrip_bitwise() {
        let config = ModelConfig::default();
        let model = TinyTransformer::new(config.clone(), 42);
        let ckpt = model.save_checkpoint();

        let mut restored = TinyTransformer::new(config, 999);
        restored.load_checkpoint(&ckpt).expect("Failed to load checkpoint");

        let tokens = vec![1, 4, 5, 3];
        let logits1 = model.forward(&tokens);
        let logits2 = restored.forward(&tokens);
        assert_eq!(logits1, logits2, "Logits must match bitwise after checkpoint load");
        assert_eq!(model.checksum(), restored.checksum());
    }

    #[test]
    fn test_finite_difference_gradients() {
        // Small miniature model to allow fast numerical verification
        let config = ModelConfig {
            vocab_size: 8,
            max_seq_len: 8,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
        };
        let mut model = TinyTransformer::new(config, 1234);

        let examples = vec![
            TaskExample {
                input_tokens: vec![1, 4, 3],
                target_tokens: vec![5, 2],
            },
        ];

        let (_, ana_grads) = model.compute_loss_and_gradients(&examples);
        let eps = 1e-3;

        // Sample indices across all parameter sections:
        // token_emb, pos_emb, w_q, w_k, w_v, w_o, attn_ln, w_1, w_2, ffn_ln, ln_f, head
        let total = model.total_parameters();
        let step = (total / 15).max(1);
        let test_indices: Vec<usize> = (0..total).step_by(step).collect();

        for &idx in &test_indices {
            let mut params = model.extract_parameters();
            let orig = params[idx];

            params[idx] = orig + eps;
            model.load_parameters(&params);
            let (loss_pos, _) = model.compute_loss_and_gradients(&examples);

            params[idx] = orig - eps;
            model.load_parameters(&params);
            let (loss_neg, _) = model.compute_loss_and_gradients(&examples);

            params[idx] = orig;
            model.load_parameters(&params);

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = ana_grads[idx];
            let diff = (num_grad - ana_grad).abs();
            let denom = num_grad.abs().max(ana_grad.abs()).max(1e-4);
            let rel_err = diff / denom;

            assert!(
                rel_err < 0.05 || diff < 1e-3,
                "Gradient mismatch at param {}: num={}, ana={}, rel_err={}",
                idx, num_grad, ana_grad, rel_err
            );
        }
    }

    #[test]
    fn test_exact_backprop_reduces_loss() {
        let config = ModelConfig::default();
        let mut model = TinyTransformer::new(config, 42);
        let plan = EditPlan {
            optimizer_type: OptimizerType::AdamW,
            base_lr: 0.02,
            ..Default::default()
        };

        let examples = vec![
            TaskExample {
                input_tokens: vec![1, 4, 3],
                target_tokens: vec![5, 2],
            },
            TaskExample {
                input_tokens: vec![1, 5, 3],
                target_tokens: vec![6, 2],
            },
        ];

        let loss_0 = model.train_step(&examples, &plan, 0, 20);
        let mut last_loss = loss_0;
        for s in 1..10 {
            last_loss = model.train_step(&examples, &plan, s, 20);
        }
        assert!(last_loss < loss_0, "Loss must decrease under exact backpropagation: start={}, end={}", loss_0, last_loss);
    }
}
