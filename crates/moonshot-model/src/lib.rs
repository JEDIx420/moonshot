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

impl ModelConfig {
    pub fn new_77k() -> Self {
        Self::default()
    }

    pub fn new_1m() -> Self {
        Self {
            vocab_size: 64,
            max_seq_len: 32,
            d_model: 256,
            n_heads: 8,
            n_layers: 2,
            d_ff: 512,
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
pub struct LoraAdapter {
    pub a_q: Vec<f32>, // rank x d_model
    pub b_q: Vec<f32>, // d_model x rank
    pub a_v: Vec<f32>, // rank x d_model
    pub b_v: Vec<f32>, // d_model x rank
    pub rank: usize,
    pub alpha: f32,
    pub d_model: usize,
}

impl LoraAdapter {
    pub fn new(d_model: usize, rank: usize, alpha: f32, rng: &mut ChaCha8Rng) -> Self {
        let scale = (1.0 / rank as f32).sqrt();
        let a_q = (0..rank * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let b_q = vec![0.0; d_model * rank];
        let a_v = (0..rank * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let b_v = vec![0.0; d_model * rank];
        Self { a_q, b_q, a_v, b_v, rank, alpha, d_model }
    }

    pub fn total_parameters(&self) -> usize {
        self.a_q.len() + self.b_q.len() + self.a_v.len() + self.b_v.len()
    }

    pub fn extract_parameters(&self) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.total_parameters());
        p.extend_from_slice(&self.a_q);
        p.extend_from_slice(&self.b_q);
        p.extend_from_slice(&self.a_v);
        p.extend_from_slice(&self.b_v);
        p
    }

    pub fn load_parameters(&mut self, p: &[f32]) {
        let mut offset = 0;
        let l_aq = self.a_q.len();
        self.a_q.copy_from_slice(&p[offset..offset + l_aq]);
        offset += l_aq;
        let l_bq = self.b_q.len();
        self.b_q.copy_from_slice(&p[offset..offset + l_bq]);
        offset += l_bq;
        let l_av = self.a_v.len();
        self.a_v.copy_from_slice(&p[offset..offset + l_av]);
        offset += l_av;
        let l_bv = self.b_v.len();
        self.b_v.copy_from_slice(&p[offset..offset + l_bv]);
    }

    pub fn apply_to_qv(&self, x: &[f32], seq_len: usize, q: &mut [f32], v: &mut [f32]) {
        let s = self.alpha / self.rank as f32;
        let r = self.rank;
        let d = self.d_model;
        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            // Q branch: q_t += s * B_q * (A_q * x_t)
            let mut h_q = vec![0.0f32; r];
            for i in 0..r {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += self.a_q[i * d + k] * x_t[k];
                }
                h_q[i] = dot;
            }
            for i in 0..d {
                let mut dot = 0.0f32;
                for k in 0..r {
                    dot += self.b_q[i * r + k] * h_q[k];
                }
                q[t * d + i] += s * dot;
            }

            // V branch: v_t += s * B_v * (A_v * x_t)
            let mut h_v = vec![0.0f32; r];
            for i in 0..r {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += self.a_v[i * d + k] * x_t[k];
                }
                h_v[i] = dot;
            }
            for i in 0..d {
                let mut dot = 0.0f32;
                for k in 0..r {
                    dot += self.b_v[i * r + k] * h_v[k];
                }
                v[t * d + i] += s * dot;
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
    #[serde(default)]
    pub adapter: Option<LoraAdapter>,
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
    // Adapter moments
    #[serde(skip)]
    pub adapter_adam_m: Vec<f32>,
    #[serde(skip)]
    pub adapter_adam_v: Vec<f32>,
    #[serde(skip)]
    pub adapter_opt_step: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProjectionMode {
    None,
    PreGradient,
    PostOptimizerDelta,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StepInstrumentation {
    pub step: usize,
    pub raw_grad_dot: f32,
    pub raw_grad_cosine: f32,
    pub step_alignment_dot: f32,
    pub step_alignment_cosine: f32,
    pub delta_norm: f32,
    pub anchor_norm: f32,
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
                adapter: None,
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
            adapter_adam_m: Vec::new(),
            adapter_adam_v: Vec::new(),
            adapter_opt_step: 0,
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
            let mut q = block.attn.w_q.forward(&h_norm, seq_len);
            let k = block.attn.w_k.forward(&h_norm, seq_len);
            let mut v = block.attn.w_v.forward(&h_norm, seq_len);

            if let Some(ref lora) = block.attn.adapter {
                lora.apply_to_qv(&h_norm, seq_len, &mut q, &mut v);
            }

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
                let mut q = block.attn.w_q.forward(&attn_ln_out, seq_len);
                let k = block.attn.w_k.forward(&attn_ln_out, seq_len);
                let mut v_val = block.attn.w_v.forward(&attn_ln_out, seq_len);

                if let Some(ref lora) = block.attn.adapter {
                    lora.apply_to_qv(&attn_ln_out, seq_len, &mut q, &mut v_val);
                }

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

    /// Parameter update with optional pre/post projection and full alignment instrumentation (Wave 5B)
    pub fn train_step_instrumented(
        &mut self,
        examples: &[TaskExample],
        anchor_examples: Option<&[TaskExample]>,
        plan: &EditPlan,
        step_idx: usize,
        total_steps: usize,
        projection_mode: ProjectionMode,
    ) -> (f32, Option<StepInstrumentation>) {
        if examples.is_empty() {
            return (0.0, None);
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

        let mut opt_anchor_grads: Option<Vec<f32>> = None;
        let mut raw_grad_dot = 0.0f32;
        let mut raw_grad_cosine = 0.0f32;
        let mut anchor_norm = 0.0f32;

        if let Some(anchors) = anchor_examples {
            if !anchors.is_empty() {
                let (_, a_grads) = self.compute_loss_and_gradients(anchors);
                let dot: f32 = all_grads.iter().zip(a_grads.iter()).map(|(&g1, &g2)| g1 * g2).sum();
                let g_norm_sq: f32 = all_grads.iter().map(|&g| g * g).sum();
                let a_norm_sq: f32 = a_grads.iter().map(|&g| g * g).sum();
                raw_grad_dot = dot;
                anchor_norm = a_norm_sq.sqrt();
                let denom = (g_norm_sq.sqrt() * anchor_norm).max(1e-8);
                raw_grad_cosine = dot / denom;

                if projection_mode == ProjectionMode::PreGradient && dot < 0.0 && a_norm_sq > 1e-8 {
                    let proj_coeff = dot / a_norm_sq;
                    for i in 0..all_grads.len() {
                        all_grads[i] -= proj_coeff * a_grads[i];
                    }
                }
                opt_anchor_grads = Some(a_grads);
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

        // Apply optimizer update to get deltas
        let mut params = self.extract_parameters();
        let n_params = params.len();
        let param_groups = self.parameter_groups();
        self.opt_step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut deltas = vec![0.0f32; n_params];
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
            deltas[i] = delta;
        }

        // Post-optimizer delta projection (Wave 5B)
        let mut step_alignment_dot = 0.0f32;
        let mut step_alignment_cosine = 0.0f32;
        let mut delta_norm = 0.0f32;

        if let Some(ref a_grads) = opt_anchor_grads {
            let dot: f32 = deltas.iter().zip(a_grads.iter()).map(|(&d, &g)| d * g).sum();
            let d_norm_sq: f32 = deltas.iter().map(|&d| d * d).sum();
            delta_norm = d_norm_sq.sqrt();
            step_alignment_dot = dot;
            let denom = (delta_norm * anchor_norm).max(1e-8);
            step_alignment_cosine = dot / denom;

            if projection_mode == ProjectionMode::PostOptimizerDelta && dot > 0.0 && anchor_norm > 1e-4 {
                let a_norm_sq = anchor_norm * anchor_norm;
                let proj_coeff = dot / a_norm_sq;
                for i in 0..deltas.len() {
                    deltas[i] -= proj_coeff * a_grads[i];
                }
                let new_dot: f32 = deltas.iter().zip(a_grads.iter()).map(|(&d, &g)| d * g).sum();
                step_alignment_dot = new_dot;
                let new_d_norm_sq: f32 = deltas.iter().map(|&d| d * d).sum();
                delta_norm = new_d_norm_sq.sqrt();
                step_alignment_cosine = new_dot / (delta_norm * anchor_norm).max(1e-8);
            }
        }

        for i in 0..n_params {
            params[i] += deltas[i];
        }
        self.load_parameters(&params);

        let instrumentation = opt_anchor_grads.map(|_| StepInstrumentation {
            step: step_idx,
            raw_grad_dot,
            raw_grad_cosine,
            step_alignment_dot,
            step_alignment_cosine,
            delta_norm,
            anchor_norm,
        });

        (avg_loss, instrumentation)
    }

    /// Wave 3 legacy wrapper: Parameter update with Orthogonal Gradient Projection (A-GEM)
    pub fn train_step_projected(
        &mut self,
        examples: &[TaskExample],
        anchor_examples: Option<&[TaskExample]>,
        plan: &EditPlan,
        step_idx: usize,
        total_steps: usize,
    ) -> f32 {
        self.train_step_instrumented(examples, anchor_examples, plan, step_idx, total_steps, ProjectionMode::PreGradient).0
    }

    // Attach modular rank-r LoRA adapter to query and value projections
    pub fn attach_lora(&mut self, rank: usize, alpha: f32, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for block in &mut self.blocks {
            block.attn.adapter = Some(LoraAdapter::new(self.config.d_model, rank, alpha, &mut rng));
        }
        let total_adap = self.total_adapter_parameters();
        self.adapter_adam_m = vec![0.0; total_adap];
        self.adapter_adam_v = vec![0.0; total_adap];
        self.adapter_opt_step = 0;
    }

    pub fn detach_lora(&mut self) {
        for block in &mut self.blocks {
            block.attn.adapter = None;
        }
        self.adapter_adam_m.clear();
        self.adapter_adam_v.clear();
        self.adapter_opt_step = 0;
    }

    pub fn total_adapter_parameters(&self) -> usize {
        let mut count = 0;
        for block in &self.blocks {
            if let Some(ref lora) = block.attn.adapter {
                count += lora.total_parameters();
            }
        }
        count
    }

    pub fn extract_adapter_parameters(&self) -> Vec<f32> {
        let mut p = Vec::new();
        for block in &self.blocks {
            if let Some(ref lora) = block.attn.adapter {
                p.extend(lora.extract_parameters());
            }
        }
        p
    }

    pub fn load_adapter_parameters(&mut self, p: &[f32]) {
        let mut offset = 0;
        for block in &mut self.blocks {
            if let Some(ref mut lora) = block.attn.adapter {
                let n = lora.total_parameters();
                lora.load_parameters(&p[offset..offset + n]);
                offset += n;
            }
        }
    }

    // Train ONLY adapter parameters using an EditPlan (supporting update-rule DSL operators)
    pub fn train_adapter_step_with_plan(&mut self, examples: &[TaskExample], plan: &EditPlan, step: usize, total_steps: usize) -> f32 {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let n_blocks = self.blocks.len();

        let mut d_aq: Vec<Vec<f32>> = Vec::with_capacity(n_blocks);
        let mut d_bq: Vec<Vec<f32>> = Vec::with_capacity(n_blocks);
        let mut d_av: Vec<Vec<f32>> = Vec::with_capacity(n_blocks);
        let mut d_bv: Vec<Vec<f32>> = Vec::with_capacity(n_blocks);

        for block in &self.blocks {
            if let Some(ref lora) = block.attn.adapter {
                d_aq.push(vec![0.0; lora.a_q.len()]);
                d_bq.push(vec![0.0; lora.b_q.len()]);
                d_av.push(vec![0.0; lora.a_v.len()]);
                d_bv.push(vec![0.0; lora.b_v.len()]);
            } else {
                return 0.0;
            }
        }

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

            struct AdapterCache {
                attn_ln_out: Vec<f32>,
                q: Vec<f32>,
                k: Vec<f32>,
                v: Vec<f32>,
                attn_weights: Vec<f32>,
                ffn_h1: Vec<f32>,
            }

            let mut block_caches = Vec::with_capacity(n_blocks);

            for block in &self.blocks {
                let (attn_ln_out, _, _) = block.attn.ln.forward(&h, seq_len);
                let mut q = block.attn.w_q.forward(&attn_ln_out, seq_len);
                let k = block.attn.w_k.forward(&attn_ln_out, seq_len);
                let mut v_val = block.attn.w_v.forward(&attn_ln_out, seq_len);

                if let Some(ref lora) = block.attn.adapter {
                    lora.apply_to_qv(&attn_ln_out, seq_len, &mut q, &mut v_val);
                }

                let (attn_out, attn_weights) = forward_multihead_attention(
                    &q, &k, &v_val, seq_len, d, self.config.n_heads
                );

                let proj = block.attn.w_o.forward(&attn_out, seq_len);
                for i in 0..seq_len * d {
                    h[i] += proj[i];
                }

                let (ffn_ln_out, _, _) = block.ffn.ln.forward(&h, seq_len);
                let ffn_h1 = block.ffn.w_1.forward(&ffn_ln_out, seq_len);
                let mut ffn_act = ffn_h1.clone();
                for val in ffn_act.iter_mut() {
                    if *val < 0.0 { *val = 0.0; }
                }
                let ffn_h2 = block.ffn.w_2.forward(&ffn_act, seq_len);
                for i in 0..seq_len * d {
                    h[i] += ffn_h2[i];
                }

                block_caches.push(AdapterCache {
                    attn_ln_out,
                    q,
                    k,
                    v: v_val,
                    attn_weights,
                    ffn_h1,
                });
            }

            let (h_final, _, _) = self.ln_f.forward(&h, seq_len);
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

            // Head backward
            let mut d_h = vec![0.0; seq_len * d];
            for t in 0..seq_len {
                let d_l_t = &d_logits[t * v..(t + 1) * v];
                for k_idx in 0..d {
                    let mut sum = 0.0f32;
                    for c in 0..v {
                        sum += d_l_t[c] * self.head.weight[c * d + k_idx];
                    }
                    d_h[t * d + k_idx] = sum;
                }
            }

            // Transformer blocks backward
            for (b_idx, block) in self.blocks.iter().enumerate().rev() {
                let cache = &block_caches[b_idx];

                // FFN backward
                let mut d_ffn_act = vec![0.0; seq_len * self.config.d_ff];
                for t in 0..seq_len {
                    let d_h_t = &d_h[t * d..(t + 1) * d];
                    for k_idx in 0..self.config.d_ff {
                        let mut sum = 0.0f32;
                        for i in 0..d {
                            sum += d_h_t[i] * block.ffn.w_2.weight[i * self.config.d_ff + k_idx];
                        }
                        d_ffn_act[t * self.config.d_ff + k_idx] = sum;
                    }
                }
                for i in 0..d_ffn_act.len() {
                    if cache.ffn_h1[i] <= 0.0 { d_ffn_act[i] = 0.0; }
                }

                let mut d_ffn_ln_out = vec![0.0; seq_len * d];
                for t in 0..seq_len {
                    let d_act_t = &d_ffn_act[t * self.config.d_ff..(t + 1) * self.config.d_ff];
                    for k_idx in 0..d {
                        let mut sum = 0.0f32;
                        for i in 0..self.config.d_ff {
                            sum += d_act_t[i] * block.ffn.w_1.weight[i * d + k_idx];
                        }
                        d_ffn_ln_out[t * d + k_idx] = sum;
                    }
                }
                for i in 0..seq_len * d {
                    d_h[i] += d_ffn_ln_out[i];
                }

                // Attention backward
                let mut d_attn_out = vec![0.0; seq_len * d];
                for t in 0..seq_len {
                    let d_proj_t = &d_h[t * d..(t + 1) * d];
                    for k_idx in 0..d {
                        let mut sum = 0.0f32;
                        for i in 0..d {
                            sum += d_proj_t[i] * block.attn.w_o.weight[i * d + k_idx];
                        }
                        d_attn_out[t * d + k_idx] = sum;
                    }
                }

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

                if let Some(ref lora) = block.attn.adapter {
                    let s = lora.alpha / lora.rank as f32;
                    let r = lora.rank;
                    for t in 0..seq_len {
                        let x_t = &cache.attn_ln_out[t * d..(t + 1) * d];
                        let d_q_t = &d_q[t * d..(t + 1) * d];
                        let d_v_t = &d_v[t * d..(t + 1) * d];

                        let mut h_q = vec![0.0f32; r];
                        for i in 0..r {
                            let mut dot = 0.0f32;
                            for m in 0..d {
                                dot += lora.a_q[i * d + m] * x_t[m];
                            }
                            h_q[i] = dot;
                        }
                        for i in 0..d {
                            for k_idx in 0..r {
                                d_bq[b_idx][i * r + k_idx] += s * d_q_t[i] * h_q[k_idx];
                            }
                        }
                        let mut d_hq = vec![0.0f32; r];
                        for k_idx in 0..r {
                            let mut dot = 0.0f32;
                            for i in 0..d {
                                dot += d_q_t[i] * lora.b_q[i * r + k_idx];
                            }
                            d_hq[k_idx] = dot;
                        }
                        for k_idx in 0..r {
                            for m in 0..d {
                                d_aq[b_idx][k_idx * d + m] += s * d_hq[k_idx] * x_t[m];
                            }
                        }

                        let mut h_v = vec![0.0f32; r];
                        for i in 0..r {
                            let mut dot = 0.0f32;
                            for m in 0..d {
                                dot += lora.a_v[i * d + m] * x_t[m];
                            }
                            h_v[i] = dot;
                        }
                        for i in 0..d {
                            for k_idx in 0..r {
                                d_bv[b_idx][i * r + k_idx] += s * d_v_t[i] * h_v[k_idx];
                            }
                        }
                        let mut d_hv = vec![0.0f32; r];
                        for k_idx in 0..r {
                            let mut dot = 0.0f32;
                            for i in 0..d {
                                dot += d_v_t[i] * lora.b_v[i * r + k_idx];
                            }
                            d_hv[k_idx] = dot;
                        }
                        for k_idx in 0..r {
                            for m in 0..d {
                                d_av[b_idx][k_idx * d + m] += s * d_hv[k_idx] * x_t[m];
                            }
                        }
                    }
                }
            }
        }

        // Apply EditPlan updates to adapter parameters
        let norm_factor = total_tokens_scored.max(1) as f32;
        let mut all_adapter_grads = Vec::new();
        for b_idx in 0..n_blocks {
            all_adapter_grads.extend(d_aq[b_idx].iter().map(|&g| g / norm_factor));
            all_adapter_grads.extend(d_bq[b_idx].iter().map(|&g| g / norm_factor));
            all_adapter_grads.extend(d_av[b_idx].iter().map(|&g| g / norm_factor));
            all_adapter_grads.extend(d_bv[b_idx].iter().map(|&g| g / norm_factor));
        }

        // Clip gradient norm if specified
        if let Some(clip) = plan.grad_clip {
            let norm_sq: f32 = all_adapter_grads.iter().map(|&g| g * g).sum();
            let norm = norm_sq.sqrt();
            if norm > clip && norm > 1e-7 {
                let scale = clip / norm;
                for g in all_adapter_grads.iter_mut() {
                    *g *= scale;
                }
            }
        }

        let mut adapter_params = self.extract_adapter_parameters();
        let total_adap = adapter_params.len();
        if self.adapter_adam_m.len() != total_adap {
            self.adapter_adam_m = vec![0.0; total_adap];
            self.adapter_adam_v = vec![0.0; total_adap];
            self.adapter_opt_step = 0;
        }

        self.adapter_opt_step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let schedule_factor = match plan.schedule_policy {
            moonshot_core::SchedulePolicy::Constant => 1.0,
            moonshot_core::SchedulePolicy::CosineDecay => {
                if total_steps > 0 {
                    0.5 * (1.0 + (std::f32::consts::PI * step as f32 / total_steps as f32).cos())
                } else {
                    1.0
                }
            }
            moonshot_core::SchedulePolicy::WarmupCosine => {
                let warmup_steps = (total_steps as f32 * 0.1).ceil() as usize;
                if step < warmup_steps && warmup_steps > 0 {
                    (step + 1) as f32 / warmup_steps as f32
                } else if total_steps > warmup_steps {
                    let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
                    0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
                } else {
                    1.0
                }
            }
        };

        let effective_lr = plan.base_lr * schedule_factor;

        for i in 0..total_adap {
            let g = all_adapter_grads[i];
            match plan.optimizer_type {
                moonshot_core::OptimizerType::Sgd => {
                    let delta = -effective_lr * (g + plan.weight_decay * adapter_params[i]);
                    adapter_params[i] += delta;
                }
                _ => {
                    self.adapter_adam_m[i] = beta1 * self.adapter_adam_m[i] + (1.0 - beta1) * g;
                    self.adapter_adam_v[i] = beta2 * self.adapter_adam_v[i] + (1.0 - beta2) * g * g;
                    let m_hat = self.adapter_adam_m[i] / (1.0 - beta1.powi(self.adapter_opt_step as i32)).max(1e-6);
                    let v_hat = self.adapter_adam_v[i] / (1.0 - beta2.powi(self.adapter_opt_step as i32)).max(1e-6);
                    let delta = -effective_lr * (m_hat / (v_hat.sqrt() + eps) + plan.weight_decay * adapter_params[i]);
                    adapter_params[i] += delta;
                }
            }
        }

        self.load_adapter_parameters(&adapter_params);

        if total_tokens_scored > 0 {
            total_loss / total_tokens_scored as f32
        } else {
            0.0
        }
    }

    // Convenience wrapper for training adapter step with default AdamW plan
    pub fn train_adapter_step(&mut self, examples: &[TaskExample], lr: f32, step: usize, total_steps: usize) -> f32 {
        let plan = EditPlan {
            base_lr: lr,
            schedule_policy: moonshot_core::SchedulePolicy::WarmupCosine,
            inner_steps: total_steps,
            ..Default::default()
        };
        self.train_adapter_step_with_plan(examples, &plan, step, total_steps)
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

    #[test]
    fn test_1m_model_parameter_count_and_gradients() {
        let config = ModelConfig::new_1m();
        let mut model = TinyTransformer::new(config, 42);
        assert_eq!(model.total_parameters(), 1_095_744, "1M Model parameter count must be exactly 1,095,744");

        let plan = EditPlan {
            optimizer_type: OptimizerType::AdamW,
            base_lr: 0.01,
            ..Default::default()
        };

        let examples = vec![
            TaskExample {
                input_tokens: vec![1, 10, 20],
                target_tokens: vec![30, 2],
            },
            TaskExample {
                input_tokens: vec![1, 15, 25],
                target_tokens: vec![35, 2],
            },
        ];

        let loss_0 = model.train_step(&examples, &plan, 0, 10);
        let mut last_loss = loss_0;
        for s in 1..5 {
            last_loss = model.train_step(&examples, &plan, s, 10);
        }
        assert!(last_loss < loss_0, "1M Model exact backprop must decrease loss: start={}, end={}", loss_0, last_loss);
    }

    #[test]
    fn test_lora_adapter_zero_init_and_isolation() {
        let config = ModelConfig::new_1m();
        let mut model = TinyTransformer::new(config, 42);
        let base_params_before = model.extract_parameters();

        let tokens = vec![1, 10, 20, 3];
        let logits_base = model.forward(&tokens);

        // Attach rank-4 LoRA adapter
        model.attach_lora(4, 8.0, 101);
        assert_eq!(model.total_adapter_parameters(), 8_192, "Rank-4 adapter across 2 layers must have 8,192 parameters");

        // Zero-init property: B=0 means adapter output is exactly zero at initialization
        let logits_with_lora = model.forward(&tokens);
        assert_eq!(logits_base, logits_with_lora, "Logits before adapter training must match base model exactly");

        let examples = vec![
            TaskExample {
                input_tokens: vec![1, 10, 20],
                target_tokens: vec![30, 2],
            },
        ];

        let loss_0 = model.train_adapter_step(&examples, 0.02, 0, 10);
        let mut last_loss = loss_0;
        for s in 1..5 {
            last_loss = model.train_adapter_step(&examples, 0.02, s, 10);
        }
        assert!(last_loss < loss_0, "Adapter loss must decrease: start={}, end={}", loss_0, last_loss);

        // Crucial parameter isolation: Base model parameters must NOT have changed by even 1 bit!
        let base_params_after = model.extract_parameters();
        assert_eq!(base_params_before, base_params_after, "Base model parameters must remain strictly frozen during adapter training");
    }
}
