use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AutodiffError {
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidDimension {
        expected: usize,
        actual: usize,
    },
    InvalidAxis {
        axis: usize,
        ndim: usize,
    },
    InvalidVar(usize),
    EmptyTensor,
    TargetOutOfBounds {
        target: usize,
        num_classes: usize,
    },
    DimensionMismatch(String),
}

impl fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutodiffError::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, actual)
            }
            AutodiffError::InvalidDimension { expected, actual } => {
                write!(
                    f,
                    "Invalid dimension: expected {}D tensor, got {}D",
                    expected, actual
                )
            }
            AutodiffError::InvalidAxis { axis, ndim } => {
                write!(
                    f,
                    "Invalid axis {} for tensor with {} dimensions",
                    axis, ndim
                )
            }
            AutodiffError::InvalidVar(id) => {
                write!(f, "Invalid variable handle: Var({})", id)
            }
            AutodiffError::EmptyTensor => {
                write!(f, "Operation cannot be performed on empty tensor")
            }
            AutodiffError::TargetOutOfBounds {
                target,
                num_classes,
            } => {
                write!(
                    f,
                    "Target index {} is out of bounds for num_classes {}",
                    target, num_classes
                )
            }
            AutodiffError::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            }
        }
    }
}

impl std::error::Error for AutodiffError {}
