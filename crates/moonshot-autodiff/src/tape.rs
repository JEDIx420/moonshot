use crate::error::AutodiffError;
use crate::ops;
use crate::tensor::Tensor;

/// Strongly-typed variable handle representing an expression node on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(pub usize);

/// Map of variable handles to their accumulated gradient tensors.
#[derive(Debug, Clone)]
pub struct Gradients {
    grads: Vec<Option<Tensor>>,
}

impl Gradients {
    pub fn new(size: usize) -> Self {
        Self {
            grads: vec![None; size],
        }
    }

    /// Retrieve the gradient for a variable, if one was computed.
    pub fn get(&self, var: Var) -> Option<&Tensor> {
        self.grads.get(var.0).and_then(|opt| opt.as_ref())
    }

    /// Retrieve the gradient for a variable, or return a zeros tensor with the given shape.
    pub fn get_or_zero(&self, var: Var, shape: &[usize]) -> Tensor {
        self.get(var).cloned().unwrap_or_else(|| Tensor::zeros(shape))
    }
}

impl std::ops::Index<Var> for Gradients {
    type Output = Tensor;

    fn index(&self, var: Var) -> &Self::Output {
        self.get(var).unwrap_or_else(|| {
            panic!(
                "No gradient available for Var({}). Ensure it participates in the backward graph.",
                var.0
            )
        })
    }
}

#[derive(Debug, Clone)]
pub enum TapeOp {
    Input,
    Add,
    Sub,
    Mul,
    MatMul,
    Gelu,
    Softmax { axis: usize },
    LayerNorm { eps: f32, means: Vec<f32>, std_invs: Vec<f32> },
    CrossEntropy { targets: Vec<usize>, probs: Vec<f32> },
    Sum,
}

#[derive(Debug, Clone)]
pub struct TapeNode {
    pub var: Var,
    pub op: TapeOp,
    pub inputs: Vec<Var>,
    pub output: Tensor,
}

/// Wengert tape recording forward computation graph for reverse-mode automatic differentiation.
#[derive(Debug, Clone, Default)]
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    /// Create a new, empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Number of nodes recorded on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tape has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all nodes from the tape.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Register a leaf tensor variable on the tape.
    pub fn var(&mut self, tensor: Tensor) -> Var {
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Input,
            inputs: Vec::new(),
            output: tensor,
        });
        var
    }

    /// Inspect the forward tensor value of a variable.
    pub fn value(&self, var: Var) -> Result<&Tensor, AutodiffError> {
        self.nodes
            .get(var.0)
            .map(|node| &node.output)
            .ok_or(AutodiffError::InvalidVar(var.0))
    }

    /// Addition: elementwise or 2D row-broadcasting
    pub fn add(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let t_b = self.value(b)?;
        let out = ops::forward_add(t_a, t_b)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Add,
            inputs: vec![a, b],
            output: out,
        });
        Ok(var)
    }

    /// Subtraction: elementwise or 2D row-broadcasting
    pub fn sub(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let t_b = self.value(b)?;
        let out = ops::forward_sub(t_a, t_b)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Sub,
            inputs: vec![a, b],
            output: out,
        });
        Ok(var)
    }

    /// Multiplication: Hadamard elementwise or 2D row-broadcasting
    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let t_b = self.value(b)?;
        let out = ops::forward_mul(t_a, t_b)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Mul,
            inputs: vec![a, b],
            output: out,
        });
        Ok(var)
    }

    /// Matrix multiplication: 2D matmul [M, K] x [K, N] -> [M, N]
    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let t_b = self.value(b)?;
        let out = ops::forward_matmul(t_a, t_b)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::MatMul,
            inputs: vec![a, b],
            output: out,
        });
        Ok(var)
    }

    /// GELU activation: Tanh approximation
    pub fn gelu(&mut self, a: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let out = ops::forward_gelu(t_a);
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Gelu,
            inputs: vec![a],
            output: out,
        });
        Ok(var)
    }

    /// Softmax: row-wise or col-wise with max subtraction for numerical stability
    pub fn softmax(&mut self, a: Var, axis: usize) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let out = ops::forward_softmax(t_a, axis)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Softmax { axis },
            inputs: vec![a],
            output: out,
        });
        Ok(var)
    }

    /// Layer normalization with analytical scale and shift adjoints
    pub fn layernorm(
        &mut self,
        a: Var,
        gamma: Var,
        beta: Var,
        eps: f32,
    ) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let t_g = self.value(gamma)?;
        let t_b = self.value(beta)?;
        let (out, means, std_invs) = ops::forward_layernorm(t_a, t_g, t_b, eps)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::LayerNorm {
                eps,
                means,
                std_invs,
            },
            inputs: vec![a, gamma, beta],
            output: out,
        });
        Ok(var)
    }

    /// Cross-Entropy loss with LogSumExp numerical stabilization
    pub fn cross_entropy(&mut self, logits: Var, targets: &[usize]) -> Result<Var, AutodiffError> {
        let t_logits = self.value(logits)?;
        let (out, probs) = ops::forward_cross_entropy(t_logits, targets)?;
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::CrossEntropy {
                targets: targets.to_vec(),
                probs,
            },
            inputs: vec![logits],
            output: out,
        });
        Ok(var)
    }

    /// Sum reduction: helper for scalar reduction
    pub fn sum(&mut self, a: Var) -> Result<Var, AutodiffError> {
        let t_a = self.value(a)?;
        let out = ops::forward_sum(t_a);
        let id = self.nodes.len();
        let var = Var(id);
        self.nodes.push(TapeNode {
            var,
            op: TapeOp::Sum,
            inputs: vec![a],
            output: out,
        });
        Ok(var)
    }

    fn accumulate_grad(
        grads: &mut [Option<Tensor>],
        var: Var,
        shape: &[usize],
        delta: &[f32],
    ) {
        if let Some(existing) = &mut grads[var.0] {
            for (g, d) in existing.data.iter_mut().zip(delta.iter()) {
                *g += *d;
            }
        } else {
            grads[var.0] = Some(Tensor {
                shape: shape.to_vec(),
                data: delta.to_vec(),
            });
        }
    }

    /// Execute reverse-mode automatic differentiation starting from `loss`.
    /// Traverses nodes in reverse topological order, accumulating fan-out gradients into `Gradients`.
    pub fn backward(&self, loss: Var) -> Result<Gradients, AutodiffError> {
        if loss.0 >= self.nodes.len() {
            return Err(AutodiffError::InvalidVar(loss.0));
        }

        let mut grads = vec![None; self.nodes.len()];
        let loss_shape = self.nodes[loss.0].output.shape.clone();
        grads[loss.0] = Some(Tensor::ones(&loss_shape));

        let mut da_buf = Vec::new();
        let mut db_buf = Vec::new();
        let mut dc_buf = Vec::new();

        for node_idx in (0..=loss.0).rev() {
            let node_grad = match &grads[node_idx] {
                Some(g) => g.clone(),
                None => continue,
            };

            let node = &self.nodes[node_idx];
            match &node.op {
                TapeOp::Input => {}
                TapeOp::Add => {
                    let a_var = node.inputs[0];
                    let b_var = node.inputs[1];
                    let a = &self.nodes[a_var.0].output;
                    let b = &self.nodes[b_var.0].output;
                    da_buf.clear();
                    db_buf.clear();
                    ops::backward_add(&node_grad, &a.shape, &b.shape, &mut da_buf, &mut db_buf)?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                    Self::accumulate_grad(&mut grads, b_var, &b.shape, &db_buf);
                }
                TapeOp::Sub => {
                    let a_var = node.inputs[0];
                    let b_var = node.inputs[1];
                    let a = &self.nodes[a_var.0].output;
                    let b = &self.nodes[b_var.0].output;
                    da_buf.clear();
                    db_buf.clear();
                    ops::backward_sub(&node_grad, &a.shape, &b.shape, &mut da_buf, &mut db_buf)?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                    Self::accumulate_grad(&mut grads, b_var, &b.shape, &db_buf);
                }
                TapeOp::Mul => {
                    let a_var = node.inputs[0];
                    let b_var = node.inputs[1];
                    let a = &self.nodes[a_var.0].output;
                    let b = &self.nodes[b_var.0].output;
                    da_buf.clear();
                    db_buf.clear();
                    ops::backward_mul(&node_grad, a, b, &mut da_buf, &mut db_buf)?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                    Self::accumulate_grad(&mut grads, b_var, &b.shape, &db_buf);
                }
                TapeOp::MatMul => {
                    let a_var = node.inputs[0];
                    let b_var = node.inputs[1];
                    let a = &self.nodes[a_var.0].output;
                    let b = &self.nodes[b_var.0].output;
                    da_buf.clear();
                    db_buf.clear();
                    ops::backward_matmul(&node_grad, a, b, &mut da_buf, &mut db_buf)?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                    Self::accumulate_grad(&mut grads, b_var, &b.shape, &db_buf);
                }
                TapeOp::Gelu => {
                    let a_var = node.inputs[0];
                    let a = &self.nodes[a_var.0].output;
                    da_buf.clear();
                    ops::backward_gelu(&node_grad, a, &mut da_buf);
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                }
                TapeOp::Softmax { axis } => {
                    let a_var = node.inputs[0];
                    let a = &self.nodes[a_var.0].output;
                    da_buf.clear();
                    ops::backward_softmax(&node_grad, &node.output, *axis, &mut da_buf)?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                }
                TapeOp::LayerNorm {
                    eps: _,
                    means,
                    std_invs,
                } => {
                    let a_var = node.inputs[0];
                    let gamma_var = node.inputs[1];
                    let beta_var = node.inputs[2];
                    let a = &self.nodes[a_var.0].output;
                    let gamma = &self.nodes[gamma_var.0].output;
                    let beta = &self.nodes[beta_var.0].output;
                    da_buf.clear();
                    db_buf.clear();
                    dc_buf.clear();
                    ops::backward_layernorm(
                        &node_grad,
                        a,
                        gamma,
                        means,
                        std_invs,
                        &mut da_buf,
                        &mut db_buf,
                        &mut dc_buf,
                    )?;
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                    Self::accumulate_grad(&mut grads, gamma_var, &gamma.shape, &db_buf);
                    Self::accumulate_grad(&mut grads, beta_var, &beta.shape, &dc_buf);
                }
                TapeOp::CrossEntropy { targets, probs } => {
                    let logits_var = node.inputs[0];
                    let logits = &self.nodes[logits_var.0].output;
                    let (b, c) = if logits.shape.len() == 2 {
                        (logits.shape[0], logits.shape[1])
                    } else {
                        (1, logits.shape[0])
                    };
                    da_buf.clear();
                    ops::backward_cross_entropy(
                        &node_grad,
                        probs,
                        targets,
                        b,
                        c,
                        &mut da_buf,
                    );
                    Self::accumulate_grad(&mut grads, logits_var, &logits.shape, &da_buf);
                }
                TapeOp::Sum => {
                    let a_var = node.inputs[0];
                    let a = &self.nodes[a_var.0].output;
                    da_buf.clear();
                    ops::backward_sum(&node_grad, a, &mut da_buf);
                    Self::accumulate_grad(&mut grads, a_var, &a.shape, &da_buf);
                }
            }
        }

        Ok(Gradients { grads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_basic_and_clear() {
        let mut tape = Tape::new();
        assert!(tape.is_empty());
        let x = tape.var(Tensor::new(&[2], vec![3.0, 4.0]).unwrap());
        assert_eq!(tape.len(), 1);
        let val = tape.value(x).unwrap();
        assert_eq!(val.data, vec![3.0, 4.0]);

        tape.clear();
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
        assert!(tape.value(x).is_err());
    }

    #[test]
    fn test_tape_error_handling() {
        let mut tape = Tape::new();
        let invalid = Var(999);
        assert!(tape.value(invalid).is_err());
        assert!(tape.add(invalid, invalid).is_err());

        // Incompatible shapes for matmul
        let a = tape.var(Tensor::new(&[2, 3], vec![1.0; 6]).unwrap());
        let b = tape.var(Tensor::new(&[4, 2], vec![1.0; 8]).unwrap());
        assert!(tape.matmul(a, b).is_err());
    }

    #[test]
    fn test_gradients_indexing() {
        let mut tape = Tape::new();
        let a = tape.var(Tensor::new(&[2], vec![2.0, -1.0]).unwrap());
        let b = tape.var(Tensor::new(&[2], vec![3.0, 5.0]).unwrap());
        let c = tape.add(a, b).unwrap();
        let loss = tape.sum(c).unwrap();

        let grads = tape.backward(loss).unwrap();
        let grad_a = &grads[a];
        assert_eq!(grad_a.data, vec![1.0, 1.0]);
        let grad_b = grads.get(b).unwrap();
        assert_eq!(grad_b.data, vec![1.0, 1.0]);

        // Non-participating var get_or_zero
        let unused = Var(999);
        let zero = grads.get_or_zero(unused, &[2]);
        assert_eq!(zero.data, vec![0.0, 0.0]);
    }

    #[test]
    fn test_cubic_fanout() {
        // y = (x * x) * x -> dy/dx = 3 * x^2
        let mut tape = Tape::new();
        let x = tape.var(Tensor::new(&[3], vec![1.0, 2.0, 3.0]).unwrap());
        let x2 = tape.mul(x, x).unwrap();
        let x3 = tape.mul(x2, x).unwrap();
        let loss = tape.sum(x3).unwrap();

        let grads = tape.backward(loss).unwrap();
        let grad_x = grads.get(x).unwrap();

        // 3 * [1^2, 2^2, 3^2] = [3, 12, 27]
        assert_eq!(grad_x.data, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_residual_branch_gradient() {
        // z = gelu(x) + x
        // dz/dx = gelu'(x) + 1
        let mut tape = Tape::new();
        let x = tape.var(Tensor::new(&[2], vec![0.0, 1.0]).unwrap());
        let gx = tape.gelu(x).unwrap();
        let z = tape.add(gx, x).unwrap();
        let loss = tape.sum(z).unwrap();

        let grads = tape.backward(loss).unwrap();
        let grad_x = grads.get(x).unwrap();

        // At x = 0, gelu(0) = 0, gelu'(0) = 0.5 -> dz/dx = 0.5 + 1 = 1.5
        assert!((grad_x.data[0] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_1d() {
        // Logits [3], target: [1]
        let mut tape = Tape::new();
        let logits = tape.var(Tensor::new(&[3], vec![2.0, 1.0, 0.1]).unwrap());
        let loss = tape.cross_entropy(logits, &[1]).unwrap();

        let grads = tape.backward(loss).unwrap();
        let dlogits = grads.get(logits).unwrap();

        // Analytical check: sum of gradients for cross-entropy with respect to logits is 0
        let sum_d: f32 = dlogits.data.iter().sum();
        assert!(sum_d.abs() < 1e-5);
    }
}
