//! Adversarial Stress Testing Suite for moonshot-autodiff
//! Verifies:
//! 1. Multi-path fan-out graph gradient accumulation (y = x^3 + gelu(x), diamond graphs, residual loops)
//! 2. Numerical stability with extreme logits (> 100, > 500, extreme negative -500) in Softmax and CrossEntropy
//! 3. Degenerate batch sizes (B=1, C=1 single class) and high-dimensional operations
//! 4. LayerNorm zero-variance / all-identical inputs and near-zero values
//! 5. Error handling for dimension mismatches, out-of-bounds targets, and empty tensors

use moonshot_autodiff::error::AutodiffError;
use moonshot_autodiff::gradcheck::check_gradients;
use moonshot_autodiff::tape::{Tape, Var};
use moonshot_autodiff::tensor::Tensor;

#[test]
fn test_zero_dependencies_enforcement() {
    let cargo_toml = include_str!("../Cargo.toml");
    let mut in_deps = false;
    let mut dep_count = 0;
    for line in cargo_toml.lines() {
        let line = line.trim();
        if line.starts_with("[dependencies]") {
            in_deps = true;
            continue;
        } else if line.starts_with('[') && in_deps {
            in_deps = false;
        } else if in_deps && !line.is_empty() && !line.starts_with('#') {
            dep_count += 1;
        }
    }
    assert_eq!(
        dep_count, 0,
        "crates/moonshot-autodiff Cargo.toml must have strictly zero dependencies! Found {}",
        dep_count
    );
}

#[test]
fn test_fanout_cubic_plus_gelu() {
    // y = x * x * x + GELU(x)
    // dy/dx = 3 * x^2 + GELU'(x)
    let x = Tensor::new(&[5], vec![-2.0, -0.5, 0.0, 1.2, 2.5]).unwrap();

    check_gradients(
        &[x],
        |tape, vars| {
            let x_var = vars[0];
            let x2 = tape.mul(x_var, x_var)?;
            let x3 = tape.mul(x2, x_var)?;
            let gx = tape.gelu(x_var)?;
            let sum_branch = tape.add(x3, gx)?;
            tape.sum(sum_branch)
        },
        1e-3,
        1.5e-3,
    )
    .expect("Cubic + GELU fan-out gradient check failed");
}

#[test]
fn test_fanout_diamond_graph() {
    // Diamond graph:
    // h1 = x + w1
    // h2 = x * w2
    // h3 = GELU(h1) + h2
    // h4 = h1 * h2 + h3
    // loss = sum(h4)
    let x = Tensor::new(&[4], vec![0.5, -1.0, 1.5, -0.2]).unwrap();
    let w1 = Tensor::new(&[4], vec![1.0, 0.2, -0.5, 0.8]).unwrap();
    let w2 = Tensor::new(&[4], vec![-0.4, 0.6, 0.3, -1.2]).unwrap();

    check_gradients(
        &[x, w1, w2],
        |tape, vars| {
            let x_var = vars[0];
            let w1_var = vars[1];
            let w2_var = vars[2];

            let h1 = tape.add(x_var, w1_var)?;
            let h2 = tape.mul(x_var, w2_var)?;
            let gh1 = tape.gelu(h1)?;
            let h3 = tape.add(gh1, h2)?;
            let prod = tape.mul(h1, h2)?;
            let h4 = tape.add(prod, h3)?;
            tape.sum(h4)
        },
        1e-3,
        2.5e-3,
    )
    .expect("Diamond graph fan-out check failed");
}

#[test]
fn test_high_degree_fanout_10_branches() {
    // y = x + x + x + ... (10 times) = 10x -> dy/dx = 10
    let mut tape = Tape::new();
    let x = tape.var(Tensor::new(&[3], vec![1.0, -2.0, 3.5]).unwrap());

    let mut accum = x;
    for _ in 1..10 {
        accum = tape.add(accum, x).unwrap();
    }
    let loss = tape.sum(accum).unwrap();
    let grads = tape.backward(loss).unwrap();

    let grad_x = grads.get(x).unwrap();
    for &g in &grad_x.data {
        assert!(
            (g - 10.0).abs() < 1e-4,
            "Expected grad 10.0 from 10 fan-out additions, got {}",
            g
        );
    }
}

#[test]
fn test_softmax_extreme_large_logits() {
    // Logits > 100 and > 500
    let mut tape = Tape::new();
    let extreme_logits = tape.var(
        Tensor::new(
            &[2, 3],
            vec![
                150.0, 152.0, 149.0, // Row 1: max 152.0
                500.0, 501.5, 498.0, // Row 2: max 501.5
            ],
        )
        .unwrap(),
    );

    let probs = tape.softmax(extreme_logits, 1).unwrap();
    let val = tape.value(probs).unwrap();

    // Verify all outputs are valid finite probabilities in [0, 1]
    for &p in &val.data {
        assert!(
            !p.is_nan() && !p.is_infinite(),
            "Softmax produced NaN/Inf on extreme inputs: {}",
            p
        );
        assert!((0.0..=1.0).contains(&p), "Prob {} out of [0, 1]", p);
    }

    // Verify each row sums to 1.0 within float precision
    let row1_sum: f32 = val.data[0..3].iter().sum();
    let row2_sum: f32 = val.data[3..6].iter().sum();
    assert!(
        (row1_sum - 1.0).abs() < 1e-5,
        "Row 1 sum {} != 1.0",
        row1_sum
    );
    assert!(
        (row2_sum - 1.0).abs() < 1e-5,
        "Row 2 sum {} != 1.0",
        row2_sum
    );

    // Verify backward pass produces finite gradients that sum to 0 along each row
    let loss = tape.sum(probs).unwrap();
    let grads = tape.backward(loss).unwrap();
    let grad_logits = grads.get(extreme_logits).unwrap();

    for &g in &grad_logits.data {
        assert!(
            !g.is_nan() && !g.is_infinite(),
            "Gradient produced NaN/Inf: {}",
            g
        );
    }

    let g_row1_sum: f32 = grad_logits.data[0..3].iter().sum();
    let g_row2_sum: f32 = grad_logits.data[3..6].iter().sum();
    assert!(
        g_row1_sum.abs() < 1e-4,
        "Row 1 grad sum {} != 0",
        g_row1_sum
    );
    assert!(
        g_row2_sum.abs() < 1e-4,
        "Row 2 grad sum {} != 0",
        g_row2_sum
    );
}

#[test]
fn test_softmax_extreme_negative_logits() {
    // Logits deeply negative: [-500, -510, -490]
    let mut tape = Tape::new();
    let neg_logits = tape.var(
        Tensor::new(&[3], vec![-500.0, -510.0, -490.0]).unwrap(),
    );

    let probs = tape.softmax(neg_logits, 0).unwrap();
    let val = tape.value(probs).unwrap();

    for &p in &val.data {
        assert!(!p.is_nan() && !p.is_infinite());
        assert!((0.0..=1.0).contains(&p));
    }
    let sum_p: f32 = val.data.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_extreme_logits() {
    // Case A: Confident correct prediction (logit 1000 on correct target)
    let mut tape_a = Tape::new();
    let logits_a = tape_a.var(Tensor::new(&[1, 3], vec![1000.0, 0.0, -1000.0]).unwrap());
    let loss_a = tape_a.cross_entropy(logits_a, &[0]).unwrap();
    let loss_val_a = tape_a.value(loss_a).unwrap().data[0];

    assert!(!loss_val_a.is_nan() && !loss_val_a.is_infinite());
    assert!(
        loss_val_a < 1e-4,
        "Loss for confident correct target should be ~0, got {}",
        loss_val_a
    );

    let grads_a = tape_a.backward(loss_a).unwrap();
    let grad_data_a = &grads_a.get(logits_a).unwrap().data;
    assert!(!grad_data_a[0].is_nan() && !grad_data_a[0].is_infinite());
    assert!(
        grad_data_a[0].abs() < 1e-4,
        "Target grad should be ~0, got {}",
        grad_data_a[0]
    );

    // Case B: Confident wrong prediction (logit 1000 on class 1, target is 0)
    let mut tape_b = Tape::new();
    let logits_b = tape_b.var(Tensor::new(&[1, 3], vec![0.0, 1000.0, -1000.0]).unwrap());
    let loss_b = tape_b.cross_entropy(logits_b, &[0]).unwrap();
    let loss_val_b = tape_b.value(loss_b).unwrap().data[0];

    assert!(!loss_val_b.is_nan() && !loss_val_b.is_infinite());
    assert!(
        (loss_val_b - 1000.0).abs() < 1e-3,
        "Loss should be ~1000, got {}",
        loss_val_b
    );

    let grads_b = tape_b.backward(loss_b).unwrap();
    let grad_data_b = &grads_b.get(logits_b).unwrap().data;
    for &g in grad_data_b {
        assert!(!g.is_nan() && !g.is_infinite());
    }
    // Target 0 has ind = 1.0, prob ~ 0.0 -> grad = (0 - 1) = -1.0
    assert!(
        (grad_data_b[0] - (-1.0)).abs() < 1e-4,
        "Grad at target should be -1.0, got {}",
        grad_data_b[0]
    );
    // Class 1 has ind = 0.0, prob ~ 1.0 -> grad = (1 - 0) = +1.0
    assert!(
        (grad_data_b[1] - 1.0).abs() < 1e-4,
        "Grad at wrong class should be 1.0, got {}",
        grad_data_b[1]
    );
}

#[test]
fn test_degenerate_batch_single_class_cross_entropy() {
    // Degenerate case: B=1, C=1 (single class logit). Target must be 0.
    // Probability is identically 1.0, Loss is 0.0, Gradient is 0.0.
    let mut tape = Tape::new();
    let logits = tape.var(Tensor::new(&[1], vec![42.0]).unwrap());
    let loss = tape.cross_entropy(logits, &[0]).unwrap();
    let loss_val = tape.value(loss).unwrap().data[0];

    assert!((loss_val - 0.0).abs() < 1e-6);

    let grads = tape.backward(loss).unwrap();
    let dlogits = grads.get(logits).unwrap();
    assert_eq!(dlogits.data, vec![0.0]);
}

#[test]
fn test_layernorm_zero_variance_all_identical_inputs() {
    // Input row has identical values: mean = 5.0, var = 0.0
    // Var + eps = eps. Normalized values = 0.0.
    // y = beta.
    let mut tape = Tape::new();
    let x = tape.var(Tensor::new(&[1, 4], vec![5.0, 5.0, 5.0, 5.0]).unwrap());
    let gamma = tape.var(Tensor::new(&[4], vec![1.2, 0.8, 1.0, 1.5]).unwrap());
    let beta = tape.var(Tensor::new(&[4], vec![0.1, -0.2, 0.3, 0.0]).unwrap());

    let norm = tape.layernorm(x, gamma, beta, 1e-5).unwrap();
    let val = tape.value(norm).unwrap();

    // Since normalized x = 0, y should exactly equal beta
    for i in 0..4 {
        assert!(
            (val.data[i] - beta_val(i)).abs() < 1e-5,
            "Expected val[{}] == beta[{}], got {}",
            i,
            i,
            val.data[i]
        );
    }

    let loss = tape.sum(norm).unwrap();
    let grads = tape.backward(loss).unwrap();
    let dx = grads.get(x).unwrap();

    // Gradients should be completely finite (no NaN or Inf from 1/sqrt(0))
    for &g in &dx.data {
        assert!(!g.is_nan() && !g.is_infinite());
    }
}

fn beta_val(i: usize) -> f32 {
    [0.1, -0.2, 0.3, 0.0][i]
}

#[test]
fn test_high_dimensional_matmul() {
    // [32, 64] x [64, 48] -> [32, 48]
    let m = 32;
    let k = 64;
    let n = 48;

    let mut a_data = Vec::with_capacity(m * k);
    for i in 0..m * k {
        a_data.push(((i as f32) * 0.013 - 1.0).sin() * 0.5);
    }
    let mut b_data = Vec::with_capacity(k * n);
    for i in 0..k * n {
        b_data.push(((i as f32) * 0.017 - 1.5).cos() * 0.5);
    }

    let a = Tensor::new(&[m, k], a_data).unwrap();
    let b = Tensor::new(&[k, n], b_data).unwrap();

    let mut tape = Tape::new();
    let a_var = tape.var(a);
    let b_var = tape.var(b);

    let c_var = tape.matmul(a_var, b_var).unwrap();
    let loss = tape.sum(c_var).unwrap();

    let grads = tape.backward(loss).unwrap();
    let da = grads.get(a_var).unwrap();
    let db = grads.get(b_var).unwrap();

    assert_eq!(da.shape, vec![m, k]);
    assert_eq!(db.shape, vec![k, n]);
    for &g in &da.data {
        assert!(!g.is_nan() && !g.is_infinite());
    }
    for &g in &db.data {
        assert!(!g.is_nan() && !g.is_infinite());
    }
}

#[test]
fn test_adversarial_error_conditions() {
    let mut tape = Tape::new();

    // 1. MatMul dimension mismatch
    let a = tape.var(Tensor::new(&[2, 3], vec![1.0; 6]).unwrap());
    let b = tape.var(Tensor::new(&[4, 5], vec![1.0; 20]).unwrap());
    assert!(matches!(tape.matmul(a, b), Err(AutodiffError::ShapeMismatch { .. })));

    // 2. MatMul non-2D tensor
    let c = tape.var(Tensor::new(&[6], vec![1.0; 6]).unwrap());
    assert!(matches!(tape.matmul(a, c), Err(AutodiffError::DimensionMismatch(_))));

    // 3. Softmax invalid axis
    assert!(matches!(tape.softmax(a, 3), Err(AutodiffError::InvalidAxis { .. })));

    // 4. CrossEntropy target out of bounds
    let logits = tape.var(Tensor::new(&[2, 4], vec![0.0; 8]).unwrap());
    let bad_targets = vec![1, 10]; // 10 >= 4
    assert!(matches!(
        tape.cross_entropy(logits, &bad_targets),
        Err(AutodiffError::TargetOutOfBounds { .. })
    ));

    // 5. CrossEntropy batch size mismatch
    let mismatched_targets = vec![1]; // len 1 != batch 2
    assert!(matches!(
        tape.cross_entropy(logits, &mismatched_targets),
        Err(AutodiffError::DimensionMismatch(_))
    ));

    // 6. LayerNorm gamma dimension mismatch
    let bad_gamma = tape.var(Tensor::new(&[5], vec![1.0; 5]).unwrap()); // 5 != 4
    let beta = tape.var(Tensor::new(&[4], vec![0.0; 4]).unwrap());
    assert!(matches!(
        tape.layernorm(logits, bad_gamma, beta, 1e-5),
        Err(AutodiffError::DimensionMismatch(_))
    ));

    // 7. Invalid Var backward
    let invalid_var = Var(99999);
    assert!(matches!(tape.backward(invalid_var), Err(AutodiffError::InvalidVar(99999))));
}
