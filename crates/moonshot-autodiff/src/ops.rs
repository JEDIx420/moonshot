#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::error::AutodiffError;
use crate::tensor::Tensor;

pub const GELU_COEFF: f32 = 0.044715;
pub const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2 / pi)

/// 1. Addition: elementwise or 2D row-broadcasting
pub fn forward_add(a: &Tensor, b: &Tensor) -> Result<Tensor, AutodiffError> {
    if a.shape == b.shape {
        let mut out = a.data.clone();
        for (o, bv) in out.iter_mut().zip(b.data.iter()) {
            *o += *bv;
        }
        Ok(Tensor {
            shape: a.shape.clone(),
            data: out,
        })
    } else if a.shape.len() == 2 && (b.shape == vec![a.shape[1]] || b.shape == vec![1, a.shape[1]]) {
        // Row-broadcast B across rows of A: A: [M, N], B: [N]
        let m = a.shape[0];
        let n = a.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let a_row = &a.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a_row[j] + b.data[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else if b.shape.len() == 2 && (a.shape == vec![b.shape[1]] || a.shape == vec![1, b.shape[1]]) {
        // Row-broadcast A across rows of B: A: [N], B: [M, N]
        let m = b.shape[0];
        let n = b.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let b_row = &b.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a.data[j] + b_row[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else {
        Err(AutodiffError::ShapeMismatch {
            expected: a.shape.clone(),
            actual: b.shape.clone(),
        })
    }
}

pub fn backward_add(
    dy: &Tensor,
    a_shape: &[usize],
    b_shape: &[usize],
    da_out: &mut Vec<f32>,
    db_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    if a_shape == b_shape {
        da_out.extend_from_slice(&dy.data);
        db_out.extend_from_slice(&dy.data);
    } else if a_shape.len() == 2 && (b_shape == [a_shape[1]] || b_shape == [1, a_shape[1]]) {
        da_out.extend_from_slice(&dy.data);
        let m = a_shape[0];
        let n = a_shape[1];
        db_out.resize(n, 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            for j in 0..n {
                db_out[j] += dy_row[j];
            }
        }
    } else if b_shape.len() == 2 && (a_shape == [b_shape[1]] || a_shape == [1, b_shape[1]]) {
        let m = b_shape[0];
        let n = b_shape[1];
        da_out.resize(n, 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            for j in 0..n {
                da_out[j] += dy_row[j];
            }
        }
        db_out.extend_from_slice(&dy.data);
    } else {
        return Err(AutodiffError::ShapeMismatch {
            expected: a_shape.to_vec(),
            actual: b_shape.to_vec(),
        });
    }
    Ok(())
}

/// 2. Subtraction: elementwise or 2D row-broadcasting
pub fn forward_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, AutodiffError> {
    if a.shape == b.shape {
        let mut out = a.data.clone();
        for (o, bv) in out.iter_mut().zip(b.data.iter()) {
            *o -= *bv;
        }
        Ok(Tensor {
            shape: a.shape.clone(),
            data: out,
        })
    } else if a.shape.len() == 2 && (b.shape == vec![a.shape[1]] || b.shape == vec![1, a.shape[1]]) {
        let m = a.shape[0];
        let n = a.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let a_row = &a.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a_row[j] - b.data[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else if b.shape.len() == 2 && (a.shape == vec![b.shape[1]] || a.shape == vec![1, b.shape[1]]) {
        let m = b.shape[0];
        let n = b.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let b_row = &b.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a.data[j] - b_row[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else {
        Err(AutodiffError::ShapeMismatch {
            expected: a.shape.clone(),
            actual: b.shape.clone(),
        })
    }
}

pub fn backward_sub(
    dy: &Tensor,
    a_shape: &[usize],
    b_shape: &[usize],
    da_out: &mut Vec<f32>,
    db_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    if a_shape == b_shape {
        da_out.extend_from_slice(&dy.data);
        db_out.resize(dy.data.len(), 0.0);
        for (d, y) in db_out.iter_mut().zip(dy.data.iter()) {
            *d = -*y;
        }
    } else if a_shape.len() == 2 && (b_shape == [a_shape[1]] || b_shape == [1, a_shape[1]]) {
        da_out.extend_from_slice(&dy.data);
        let m = a_shape[0];
        let n = a_shape[1];
        db_out.resize(n, 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            for j in 0..n {
                db_out[j] -= dy_row[j];
            }
        }
    } else if b_shape.len() == 2 && (a_shape == [b_shape[1]] || a_shape == [1, b_shape[1]]) {
        let m = b_shape[0];
        let n = b_shape[1];
        da_out.resize(n, 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            for j in 0..n {
                da_out[j] += dy_row[j];
            }
        }
        db_out.resize(dy.data.len(), 0.0);
        for (d, y) in db_out.iter_mut().zip(dy.data.iter()) {
            *d = -*y;
        }
    } else {
        return Err(AutodiffError::ShapeMismatch {
            expected: a_shape.to_vec(),
            actual: b_shape.to_vec(),
        });
    }
    Ok(())
}

/// 3. Multiplication: Hadamard elementwise (or 2D row broadcast)
pub fn forward_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, AutodiffError> {
    if a.shape == b.shape {
        let mut out = a.data.clone();
        for (o, bv) in out.iter_mut().zip(b.data.iter()) {
            *o *= *bv;
        }
        Ok(Tensor {
            shape: a.shape.clone(),
            data: out,
        })
    } else if a.shape.len() == 2 && (b.shape == vec![a.shape[1]] || b.shape == vec![1, a.shape[1]]) {
        let m = a.shape[0];
        let n = a.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let a_row = &a.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a_row[j] * b.data[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else if b.shape.len() == 2 && (a.shape == vec![b.shape[1]] || a.shape == vec![1, b.shape[1]]) {
        let m = b.shape[0];
        let n = b.shape[1];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            let b_row = &b.data[i * n..(i + 1) * n];
            for j in 0..n {
                out.push(a.data[j] * b_row[j]);
            }
        }
        Ok(Tensor {
            shape: vec![m, n],
            data: out,
        })
    } else {
        Err(AutodiffError::ShapeMismatch {
            expected: a.shape.clone(),
            actual: b.shape.clone(),
        })
    }
}

pub fn backward_mul(
    dy: &Tensor,
    a: &Tensor,
    b: &Tensor,
    da_out: &mut Vec<f32>,
    db_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    if a.shape == b.shape {
        da_out.resize(dy.data.len(), 0.0);
        db_out.resize(dy.data.len(), 0.0);
        for i in 0..dy.data.len() {
            da_out[i] = dy.data[i] * b.data[i];
            db_out[i] = dy.data[i] * a.data[i];
        }
    } else if a.shape.len() == 2 && (b.shape == [a.shape[1]] || b.shape == [1, a.shape[1]]) {
        let m = a.shape[0];
        let n = a.shape[1];
        da_out.resize(dy.data.len(), 0.0);
        db_out.resize(n, 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            let a_row = &a.data[i * n..(i + 1) * n];
            for j in 0..n {
                da_out[i * n + j] = dy_row[j] * b.data[j];
                db_out[j] += dy_row[j] * a_row[j];
            }
        }
    } else if b.shape.len() == 2 && (a.shape == [b.shape[1]] || a.shape == [1, b.shape[1]]) {
        let m = b.shape[0];
        let n = b.shape[1];
        da_out.resize(n, 0.0);
        db_out.resize(dy.data.len(), 0.0);
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            let b_row = &b.data[i * n..(i + 1) * n];
            for j in 0..n {
                da_out[j] += dy_row[j] * b_row[j];
                db_out[i * n + j] = dy_row[j] * a.data[j];
            }
        }
    } else {
        return Err(AutodiffError::ShapeMismatch {
            expected: a.shape.clone(),
            actual: b.shape.clone(),
        });
    }
    Ok(())
}

/// 4. Matrix Multiplication: 2D matmul [M, K] x [K, N] -> [M, N]
pub fn forward_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, AutodiffError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(AutodiffError::DimensionMismatch(format!(
            "Matmul requires 2D tensors, got {}D and {}D",
            a.shape.len(),
            b.shape.len()
        )));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let bk = b.shape[0];
    let n = b.shape[1];
    if k != bk {
        return Err(AutodiffError::ShapeMismatch {
            expected: vec![m, k],
            actual: vec![bk, n],
        });
    }

    let mut out = vec![0.0; m * n];
    for i in 0..m {
        let a_row = &a.data[i * k..(i + 1) * k];
        for p in 0..k {
            let a_ip = a_row[p];
            let b_row = &b.data[p * n..(p + 1) * n];
            for j in 0..n {
                out[i * n + j] += a_ip * b_row[j];
            }
        }
    }
    Ok(Tensor {
        shape: vec![m, n],
        data: out,
    })
}

pub fn backward_matmul(
    dy: &Tensor,
    a: &Tensor,
    b: &Tensor,
    da_out: &mut Vec<f32>,
    db_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    da_out.resize(m * k, 0.0);
    db_out.resize(k * n, 0.0);

    // da = dy * B^T -> da[i, p] = sum_j (dy[i, j] * B[p, j])
    for i in 0..m {
        let dy_row = &dy.data[i * n..(i + 1) * n];
        for p in 0..k {
            let b_row = &b.data[p * n..(p + 1) * n];
            let mut sum = 0.0;
            for j in 0..n {
                sum += dy_row[j] * b_row[j];
            }
            da_out[i * k + p] = sum;
        }
    }

    // db = A^T * dy -> db[p, j] = sum_i (A[i, p] * dy[i, j])
    for i in 0..m {
        let a_row = &a.data[i * k..(i + 1) * k];
        let dy_row = &dy.data[i * n..(i + 1) * n];
        for p in 0..k {
            let a_ip = a_row[p];
            for j in 0..n {
                db_out[p * n + j] += a_ip * dy_row[j];
            }
        }
    }

    Ok(())
}

/// 5. GELU activation: Tanh approximation 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
pub fn forward_gelu(a: &Tensor) -> Tensor {
    let mut out = Vec::with_capacity(a.data.len());
    for &x in &a.data {
        let u = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        let t = u.tanh();
        out.push(0.5 * x * (1.0 + t));
    }
    Tensor {
        shape: a.shape.clone(),
        data: out,
    }
}

pub fn backward_gelu(dy: &Tensor, a: &Tensor, da_out: &mut Vec<f32>) {
    da_out.resize(a.data.len(), 0.0);
    for i in 0..a.data.len() {
        let x = a.data[i];
        let u = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        let t = u.tanh();
        let du_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x);
        let dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx;
        da_out[i] = dy.data[i] * dy_dx;
    }
}

/// 6. Softmax: max subtraction for numerical stability
pub fn forward_softmax(a: &Tensor, axis: usize) -> Result<Tensor, AutodiffError> {
    if a.shape.is_empty() {
        return Err(AutodiffError::EmptyTensor);
    }
    if axis >= a.shape.len() {
        return Err(AutodiffError::InvalidAxis {
            axis,
            ndim: a.shape.len(),
        });
    }

    let mut out = vec![0.0; a.data.len()];

    if a.shape.len() == 1 {
        // 1D softmax
        let mut max_val = f32::NEG_INFINITY;
        for &v in &a.data {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum = 0.0;
        for i in 0..a.data.len() {
            let e = (a.data[i] - max_val).exp();
            out[i] = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum.max(1e-12);
        for v in &mut out {
            *v *= inv_sum;
        }
    } else if a.shape.len() == 2 && axis == 1 {
        // 2D row-wise softmax [M, N]
        let m = a.shape[0];
        let n = a.shape[1];
        for i in 0..m {
            let row = &a.data[i * n..(i + 1) * n];
            let mut max_val = f32::NEG_INFINITY;
            for &v in row {
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = 0.0;
            for j in 0..n {
                let e = (row[j] - max_val).exp();
                out[i * n + j] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum.max(1e-12);
            for j in 0..n {
                out[i * n + j] *= inv_sum;
            }
        }
    } else if a.shape.len() == 2 && axis == 0 {
        // 2D col-wise softmax [M, N]
        let m = a.shape[0];
        let n = a.shape[1];
        for j in 0..n {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..m {
                let v = a.data[i * n + j];
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = 0.0;
            for i in 0..m {
                let e = (a.data[i * n + j] - max_val).exp();
                out[i * n + j] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum.max(1e-12);
            for i in 0..m {
                out[i * n + j] *= inv_sum;
            }
        }
    } else {
        return Err(AutodiffError::DimensionMismatch(format!(
            "Softmax currently supports 1D or 2D (axis 0 or 1), got {}D with axis {}",
            a.shape.len(),
            axis
        )));
    }

    Ok(Tensor {
        shape: a.shape.clone(),
        data: out,
    })
}

pub fn backward_softmax(
    dy: &Tensor,
    cached_probs: &Tensor,
    axis: usize,
    da_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    da_out.resize(dy.data.len(), 0.0);

    if cached_probs.shape.len() == 1 {
        let p = &cached_probs.data;
        let mut dot = 0.0;
        for i in 0..p.len() {
            dot += dy.data[i] * p[i];
        }
        for i in 0..p.len() {
            da_out[i] = p[i] * (dy.data[i] - dot);
        }
    } else if cached_probs.shape.len() == 2 && axis == 1 {
        let m = cached_probs.shape[0];
        let n = cached_probs.shape[1];
        for i in 0..m {
            let dy_row = &dy.data[i * n..(i + 1) * n];
            let p_row = &cached_probs.data[i * n..(i + 1) * n];
            let mut dot = 0.0;
            for j in 0..n {
                dot += dy_row[j] * p_row[j];
            }
            for j in 0..n {
                da_out[i * n + j] = p_row[j] * (dy_row[j] - dot);
            }
        }
    } else if cached_probs.shape.len() == 2 && axis == 0 {
        let m = cached_probs.shape[0];
        let n = cached_probs.shape[1];
        for j in 0..n {
            let mut dot = 0.0;
            for i in 0..m {
                dot += dy.data[i * n + j] * cached_probs.data[i * n + j];
            }
            for i in 0..m {
                let p = cached_probs.data[i * n + j];
                da_out[i * n + j] = p * (dy.data[i * n + j] - dot);
            }
        }
    } else {
        return Err(AutodiffError::DimensionMismatch(format!(
            "Softmax backward supports 1D or 2D, got {}D",
            cached_probs.shape.len()
        )));
    }

    Ok(())
}

/// 7. LayerNorm: row-wise normalization with gamma scale and beta shift
pub fn forward_layernorm(
    a: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Vec<f32>, Vec<f32>), AutodiffError> {
    if a.shape.is_empty() {
        return Err(AutodiffError::EmptyTensor);
    }
    let d = *a.shape.last().unwrap();
    if gamma.numel() != d || beta.numel() != d {
        return Err(AutodiffError::DimensionMismatch(format!(
            "LayerNorm gamma/beta size {} does not match input last dimension {}",
            gamma.numel(),
            d
        )));
    }

    let rows = a.numel() / d;
    let mut out = vec![0.0; a.numel()];
    let mut means = Vec::with_capacity(rows);
    let mut std_invs = Vec::with_capacity(rows);

    let d_f32 = d as f32;
    for t in 0..rows {
        let row = &a.data[t * d..(t + 1) * d];
        let mean = row.iter().sum::<f32>() / d_f32;
        let var = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d_f32;
        let std_inv = 1.0 / (var + eps).sqrt();

        means.push(mean);
        std_invs.push(std_inv);

        for i in 0..d {
            let norm = (row[i] - mean) * std_inv;
            out[t * d + i] = norm * gamma.data[i] + beta.data[i];
        }
    }

    Ok((
        Tensor {
            shape: a.shape.clone(),
            data: out,
        },
        means,
        std_invs,
    ))
}

pub fn backward_layernorm(
    dy: &Tensor,
    a: &Tensor,
    gamma: &Tensor,
    means: &[f32],
    std_invs: &[f32],
    da_out: &mut Vec<f32>,
    dgamma_out: &mut Vec<f32>,
    dbeta_out: &mut Vec<f32>,
) -> Result<(), AutodiffError> {
    let d = *a.shape.last().unwrap();
    let rows = a.numel() / d;
    let d_f32 = d as f32;

    da_out.resize(a.numel(), 0.0);
    dgamma_out.resize(d, 0.0);
    dbeta_out.resize(d, 0.0);

    let mut dnorm = vec![0.0; d];

    for t in 0..rows {
        let mean = means[t];
        let std_inv = std_invs[t];
        let row_x = &a.data[t * d..(t + 1) * d];
        let row_dy = &dy.data[t * d..(t + 1) * d];

        for i in 0..d {
            let norm_i = (row_x[i] - mean) * std_inv;
            dgamma_out[i] += row_dy[i] * norm_i;
            dbeta_out[i] += row_dy[i];
            dnorm[i] = row_dy[i] * gamma.data[i];
        }

        let sum_dnorm: f32 = dnorm.iter().sum();
        let mut sum_dnorm_norm: f32 = 0.0;
        for i in 0..d {
            let norm_i = (row_x[i] - mean) * std_inv;
            sum_dnorm_norm += dnorm[i] * norm_i;
        }

        for i in 0..d {
            let norm_i = (row_x[i] - mean) * std_inv;
            da_out[t * d + i] = std_inv * (dnorm[i] - (sum_dnorm + norm_i * sum_dnorm_norm) / d_f32);
        }
    }

    Ok(())
}

/// 8. Cross-Entropy Loss: numerically stable LogSumExp formulation
pub fn forward_cross_entropy(
    logits: &Tensor,
    targets: &[usize],
) -> Result<(Tensor, Vec<f32>), AutodiffError> {
    let (b, c) = if logits.shape.len() == 2 {
        (logits.shape[0], logits.shape[1])
    } else if logits.shape.len() == 1 {
        (1, logits.shape[0])
    } else {
        return Err(AutodiffError::DimensionMismatch(format!(
            "Cross entropy requires 1D or 2D logits, got {}D",
            logits.shape.len()
        )));
    };

    if targets.len() != b {
        return Err(AutodiffError::DimensionMismatch(format!(
            "Cross entropy targets length {} does not match batch size {}",
            targets.len(),
            b
        )));
    }

    for &t in targets.iter() {
        if t >= c {
            return Err(AutodiffError::TargetOutOfBounds {
                target: t,
                num_classes: c,
            });
        }
    }

    let mut total_loss = 0.0;
    let mut probs = vec![0.0; b * c];

    for i in 0..b {
        let row = &logits.data[i * c..(i + 1) * c];
        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0;
        for j in 0..c {
            let e = (row[j] - max_val).exp();
            probs[i * c + j] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for j in 0..c {
            probs[i * c + j] *= inv_sum;
        }

        let target_tok = targets[i];
        let log_sum_exp = max_val + sum_exp.ln();
        let sample_loss = log_sum_exp - row[target_tok];
        total_loss += sample_loss;
    }

    let mean_loss = total_loss / (b as f32);
    Ok((
        Tensor {
            shape: vec![1],
            data: vec![mean_loss],
        },
        probs,
    ))
}

pub fn backward_cross_entropy(
    dy: &Tensor,
    probs: &[f32],
    targets: &[usize],
    b: usize,
    c: usize,
    dlogits_out: &mut Vec<f32>,
) {
    dlogits_out.resize(b * c, 0.0);
    let scale = dy.data.first().copied().unwrap_or(1.0) / (b as f32);

    for i in 0..b {
        let target_tok = targets[i];
        for j in 0..c {
            let target_ind = if j == target_tok { 1.0 } else { 0.0 };
            dlogits_out[i * c + j] = scale * (probs[i * c + j] - target_ind);
        }
    }
}

/// 9. Sum reduction: helper for scalar reduction
pub fn forward_sum(a: &Tensor) -> Tensor {
    let s: f64 = a.data.iter().map(|&x| x as f64).sum();
    Tensor {
        shape: vec![1],
        data: vec![s as f32],
    }
}

pub fn backward_sum(dy: &Tensor, a: &Tensor, da_out: &mut Vec<f32>) {
    let scale = dy.data.first().copied().unwrap_or(1.0);
    da_out.resize(a.data.len(), scale);
}
