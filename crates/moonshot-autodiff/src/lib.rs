pub mod error;
pub mod gradcheck;
pub mod ops;
pub mod tape;
pub mod tensor;

pub use error::AutodiffError;
pub use gradcheck::check_gradients;
pub use tape::{Gradients, Tape, TapeNode, TapeOp, Var};
pub use tensor::Tensor;
