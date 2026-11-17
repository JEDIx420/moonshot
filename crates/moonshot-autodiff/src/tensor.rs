use crate::error::AutodiffError;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor with the given shape and data.
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self, AutodiffError> {
        let expected_numel = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected_numel {
            return Err(AutodiffError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![data.len()],
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; numel],
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let numel = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        Self {
            shape: shape.to_vec(),
            data: vec![1.0; numel],
        }
    }

    /// Create a tensor from a slice of floats.
    pub fn from_slice(shape: &[usize], slice: &[f32]) -> Result<Self, AutodiffError> {
        Self::new(shape, slice.to_vec())
    }

    /// Number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions of the tensor.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape tensor into a new shape with identical element count.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, AutodiffError> {
        let new_numel = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        if new_numel != self.numel() {
            return Err(AutodiffError::ShapeMismatch {
                expected: new_shape.to_vec(),
                actual: self.shape.clone(),
            });
        }
        Ok(Self {
            shape: new_shape.to_vec(),
            data: self.data.clone(),
        })
    }

    /// Transpose a 2D tensor [M, N] -> [N, M].
    pub fn transpose_2d(&self) -> Result<Self, AutodiffError> {
        if self.shape.len() != 2 {
            return Err(AutodiffError::InvalidDimension {
                expected: 2,
                actual: self.shape.len(),
            });
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let mut transposed_data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                transposed_data[j * m + i] = self.data[i * n + j];
            }
        }
        Ok(Self {
            shape: vec![n, m],
            data: transposed_data,
        })
    }

    /// Compute linear flat index from multi-dimensional coordinates.
    pub fn linear_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            if indices[i] >= self.shape[i] {
                return None;
            }
            idx += indices[i] * stride;
            stride *= self.shape[i];
        }
        Some(idx)
    }

    /// Get value at multi-dimensional coordinates.
    pub fn at(&self, indices: &[usize]) -> Option<f32> {
        let idx = self.linear_index(indices)?;
        self.data.get(idx).copied()
    }

    /// Get mutable reference to value at multi-dimensional coordinates.
    pub fn at_mut(&mut self, indices: &[usize]) -> Option<&mut f32> {
        let idx = self.linear_index(indices)?;
        self.data.get_mut(idx)
    }

    /// Fast 2D element getter.
    #[inline]
    pub fn get_2d(&self, r: usize, c: usize) -> Option<f32> {
        if self.shape.len() != 2 || r >= self.shape[0] || c >= self.shape[1] {
            return None;
        }
        Some(self.data[r * self.shape[1] + c])
    }

    /// Fast 2D element setter.
    #[inline]
    pub fn set_2d(&mut self, r: usize, c: usize, val: f32) -> Result<(), AutodiffError> {
        if self.shape.len() != 2 {
            return Err(AutodiffError::InvalidDimension {
                expected: 2,
                actual: self.shape.len(),
            });
        }
        if r >= self.shape[0] || c >= self.shape[1] {
            return Err(AutodiffError::DimensionMismatch(format!(
                "Index ({}, {}) out of bounds for shape [{}, {}]",
                r, c, self.shape[0], self.shape[1]
            )));
        }
        let n = self.shape[1];
        self.data[r * n + c] = val;
        Ok(())
    }

    /// In-place elementwise addition.
    pub fn add_assign(&mut self, other: &Tensor) -> Result<(), AutodiffError> {
        if self.shape != other.shape {
            return Err(AutodiffError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
        Ok(())
    }

    /// In-place elementwise subtraction.
    pub fn sub_assign(&mut self, other: &Tensor) -> Result<(), AutodiffError> {
        if self.shape != other.shape {
            return Err(AutodiffError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
        Ok(())
    }

    /// Multiply all elements in-place by a scalar.
    pub fn scale(&mut self, factor: f32) {
        for v in self.data.iter_mut() {
            *v *= factor;
        }
    }

    /// Return a clone of raw float data.
    pub fn clone_data(&self) -> Vec<f32> {
        self.data.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_and_indexing() {
        let t = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.get_2d(0, 1), Some(2.0));
        assert_eq!(t.get_2d(1, 2), Some(6.0));
        assert_eq!(t.get_2d(2, 0), None);
        assert_eq!(t.at(&[1, 0]), Some(4.0));
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = t.transpose_2d().unwrap();
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::zeros(&[2, 3]);
        let reshaped = t.reshape(&[6]).unwrap();
        assert_eq!(reshaped.shape, vec![6]);
        assert_eq!(reshaped.numel(), 6);
    }
}
