use crate::error::AutodiffError;
use crate::tape::{Tape, Var};
use crate::tensor::Tensor;

/// Verify analytical autodiff gradients against numerical central finite differences:
///   (f(x + eps) - f(x - eps)) / (2 * eps)
/// Relative error metric:
///   |g_num - g_auto| / max(|g_num|, |g_auto|, 1e-4) < tol
pub fn check_gradients<F>(
    inputs: &[Tensor],
    build_graph: F,
    eps: f32,
    tol: f32,
) -> Result<(), String>
where
    F: Fn(&mut Tape, &[Var]) -> Result<Var, AutodiffError>,
{
    // 1. Analytical forward and backward pass
    let mut tape = Tape::new();
    let mut input_vars = Vec::with_capacity(inputs.len());
    for inp in inputs {
        input_vars.push(tape.var(inp.clone()));
    }

    let loss_var = build_graph(&mut tape, &input_vars)
        .map_err(|e| format!("Build graph forward failed: {:?}", e))?;
    let grads = tape
        .backward(loss_var)
        .map_err(|e| format!("Autodiff backward failed: {:?}", e))?;

    // 2. Numerical finite difference check for each input tensor
    for p in 0..inputs.len() {
        let ana_grad = grads.get(input_vars[p]).ok_or_else(|| {
            format!("Variable {} has no gradient in backward graph", p)
        })?;

        let numel = inputs[p].numel();
        for k in 0..numel {
            // f(theta + eps)
            let mut inputs_pos = inputs.to_vec();
            inputs_pos[p].data[k] += eps;
            let mut tape_pos = Tape::new();
            let vars_pos: Vec<Var> = inputs_pos
                .iter()
                .map(|t| tape_pos.var(t.clone()))
                .collect();
            let loss_pos = build_graph(&mut tape_pos, &vars_pos)
                .map_err(|e| format!("Pos forward failed: {:?}", e))?;
            let val_pos = tape_pos
                .value(loss_pos)
                .map_err(|e| format!("Pos value failed: {:?}", e))?
                .data[0];

            // f(theta - eps)
            let mut inputs_neg = inputs.to_vec();
            inputs_neg[p].data[k] -= eps;
            let mut tape_neg = Tape::new();
            let vars_neg: Vec<Var> = inputs_neg
                .iter()
                .map(|t| tape_neg.var(t.clone()))
                .collect();
            let loss_neg = build_graph(&mut tape_neg, &vars_neg)
                .map_err(|e| format!("Neg forward failed: {:?}", e))?;
            let val_neg = tape_neg
                .value(loss_neg)
                .map_err(|e| format!("Neg value failed: {:?}", e))?
                .data[0];

            let num_grad = (val_pos - val_neg) / (2.0 * eps);
            let ana = ana_grad.data[k];
            let diff = (num_grad - ana).abs();
            let denom = num_grad.abs().max(ana.abs()).max(1e-4);
            let rel_err = diff / denom;

            if rel_err >= tol && diff >= 1e-4 {
                return Err(format!(
                    "Gradcheck failed on input {} elem {}: num_grad={}, ana_grad={}, diff={}, rel_err={} (tol={})",
                    p, k, num_grad, ana, diff, rel_err, tol
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_add_elementwise() {
        let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(&[2, 3], vec![0.5, -1.0, 2.5, 0.0, 3.0, -2.0]).unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let sum = tape.add(vars[0], vars[1])?;
                tape.sum(sum)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_add_row_broadcast() {
        let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(&[3], vec![0.5, -1.0, 2.5]).unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let sum = tape.add(vars[0], vars[1])?;
                tape.sum(sum)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_sub_elementwise() {
        let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(&[2, 3], vec![0.5, -1.0, 2.5, 0.0, 3.0, -2.0]).unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let diff = tape.sub(vars[0], vars[1])?;
                tape.sum(diff)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_sub_row_broadcast() {
        let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(&[3], vec![0.5, -1.0, 2.5]).unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let diff = tape.sub(vars[0], vars[1])?;
                tape.sum(diff)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_mul_elementwise() {
        let a = Tensor::new(&[2, 3], vec![0.5, -0.8, 0.3, 0.4, -0.2, 0.6]).unwrap();
        let b = Tensor::new(&[2, 3], vec![0.2, 0.5, -0.4, 0.8, -0.6, 0.3]).unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let prod = tape.mul(vars[0], vars[1])?;
                tape.sum(prod)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_mul_self_fanout() {
        // y = x * x -> dy/dx = 2x
        let x = Tensor::new(&[4], vec![1.5, -2.0, 0.5, 3.0]).unwrap();

        check_gradients(
            &[x],
            |tape, vars| {
                let sq = tape.mul(vars[0], vars[0])?;
                tape.sum(sq)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_matmul() {
        let a = Tensor::new(&[2, 3], vec![1.0, -0.5, 2.0, 0.0, 1.5, -1.0]).unwrap();
        let b = Tensor::new(
            &[3, 2],
            vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.5],
        )
        .unwrap();

        check_gradients(
            &[a, b],
            |tape, vars| {
                let mm = tape.matmul(vars[0], vars[1])?;
                tape.sum(mm)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_gelu() {
        let x = Tensor::new(&[6], vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.5]).unwrap();

        check_gradients(
            &[x],
            |tape, vars| {
                let g = tape.gelu(vars[0])?;
                tape.sum(g)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_softmax_1d() {
        let x = Tensor::new(&[4], vec![1.0, 2.0, -0.5, 0.2]).unwrap();

        // Testing softmax with a linear projection weights
        let weights = Tensor::new(&[4], vec![0.5, -0.2, 1.2, 0.1]).unwrap();

        check_gradients(
            &[x, weights],
            |tape, vars| {
                let p = tape.softmax(vars[0], 0)?;
                let prod = tape.mul(p, vars[1])?;
                tape.sum(prod)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_softmax_2d_rowwise() {
        let x = Tensor::new(
            &[2, 3],
            vec![1.0, -0.5, 2.0, 0.5, 1.5, -1.0],
        )
        .unwrap();
        let weights = Tensor::new(
            &[2, 3],
            vec![0.2, -0.4, 0.8, -0.1, 0.5, 0.3],
        )
        .unwrap();

        check_gradients(
            &[x, weights],
            |tape, vars| {
                let p = tape.softmax(vars[0], 1)?;
                let prod = tape.mul(p, vars[1])?;
                tape.sum(prod)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_layernorm() {
        let x = Tensor::new(
            &[2, 4],
            vec![1.0, -1.0, 2.0, 0.5, 0.2, 1.5, -0.5, 0.8],
        )
        .unwrap();
        let gamma = Tensor::new(&[4], vec![1.2, 0.8, 1.5, 0.9]).unwrap();
        let beta = Tensor::new(&[4], vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let weights = Tensor::new(
            &[2, 4],
            vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.2, -0.3],
        )
        .unwrap();

        check_gradients(
            &[x, gamma, beta, weights],
            |tape, vars| {
                let norm = tape.layernorm(vars[0], vars[1], vars[2], 1e-5)?;
                let prod = tape.mul(norm, vars[3])?;
                tape.sum(prod)
            },
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_grad_cross_entropy() {
        let logits = Tensor::new(
            &[3, 4],
            vec![
                1.5, 0.2, -0.5, 2.0,
                -1.0, 2.5, 0.0, 0.8,
                0.5, 1.0, 2.2, -0.3,
            ],
        )
        .unwrap();
        let targets = vec![3, 1, 2];

        check_gradients(
            &[logits],
            |tape, vars| tape.cross_entropy(vars[0], &targets),
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn test_composite_transformer_block() {
        // Full pipeline:
        // Input X [2, 4]
        // -> LayerNorm(X, gamma, beta)
        // -> MatMul(W1 [4, 8]) + b1 [8]
        // -> GELU
        // -> MatMul(W2 [8, 4]) + b2 [4]
        // -> Residual Add(X)
        // -> CrossEntropy(targets: [2, 1])
        let x = Tensor::new(
            &[2, 4],
            vec![0.5, -0.2, 1.2, 0.1, -0.8, 0.6, 0.3, -0.4],
        )
        .unwrap();
        let gamma = Tensor::new(&[4], vec![1.0, 1.1, 0.9, 1.05]).unwrap();
        let beta = Tensor::new(&[4], vec![0.0, 0.1, -0.1, 0.05]).unwrap();

        // W1 [4, 8]
        let mut w1_data = Vec::with_capacity(32);
        for i in 0..32 {
            w1_data.push(((i as f32) * 0.17 - 2.5).sin() * 0.5);
        }
        let w1 = Tensor::new(&[4, 8], w1_data).unwrap();
        let b1 = Tensor::new(&[8], vec![0.1, -0.1, 0.2, 0.0, -0.2, 0.15, -0.05, 0.1]).unwrap();

        // W2 [8, 4]
        let mut w2_data = Vec::with_capacity(32);
        for i in 0..32 {
            w2_data.push(((i as f32) * 0.23 - 1.8).cos() * 0.5);
        }
        let w2 = Tensor::new(&[8, 4], w2_data).unwrap();
        let b2 = Tensor::new(&[4], vec![-0.1, 0.2, 0.05, -0.15]).unwrap();

        let targets = vec![2, 1];

        check_gradients(
            &[x, gamma, beta, w1, b1, w2, b2],
            |tape, vars| {
                let norm = tape.layernorm(vars[0], vars[1], vars[2], 1e-5)?;
                let h1 = tape.matmul(norm, vars[3])?;
                let h1_bias = tape.add(h1, vars[4])?;
                let act = tape.gelu(h1_bias)?;
                let h2 = tape.matmul(act, vars[5])?;
                let h2_bias = tape.add(h2, vars[6])?;
                let res = tape.add(h2_bias, vars[0])?; // Residual connection to X (vars[0])
                tape.cross_entropy(res, &targets)
            },
            1e-3,
            1.5e-2,
        )
        .unwrap();
    }
}
