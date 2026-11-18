use burn::module::{Module, Param};
use burn::nn::{Embedding, LayerNorm, LayerNormConfig, Linear};
use burn::tensor::{activation, Bool, Device, Int, Tensor, TensorData};
use moonshot_model::{LayerNormParam, LinearParam, TinyTransformer};

#[derive(Module, Debug)]
pub struct Model {
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
}

#[derive(Module, Debug)]
struct Block {
    query: Linear,
    key: Linear,
    value: Linear,
    project: Linear,
    attention_norm: LayerNorm,
    expand: Linear,
    contract: Linear,
    mlp_norm: LayerNorm,
}

fn parameter<const D: usize>(
    values: &[f32],
    shape: [usize; D],
    device: &Device,
) -> Param<Tensor<D>> {
    Param::from_data(TensorData::new(values.to_vec(), shape), device)
}

fn linear(source: &LinearParam, device: &Device) -> Linear {
    let weights = Tensor::<2>::from_data(
        TensorData::new(source.weight.clone(), [source.out_dim, source.in_dim]),
        device,
    );
    Linear {
        weight: Param::from_tensor(weights.transpose()),
        bias: Some(parameter(&source.bias, [source.out_dim], device)),
    }
}

fn norm(source: &LayerNormParam, device: &Device) -> LayerNorm {
    let mut layer = LayerNormConfig::new(source.dim).init(device);
    layer.gamma = parameter(&source.gamma, [source.dim], device);
    layer.beta = Some(parameter(&source.beta, [source.dim], device));
    layer
}

impl Model {
    pub fn from_moonshot(source: &TinyTransformer, device: &Device) -> Self {
        let config = &source.config;
        Self {
            tokens: Embedding {
                weight: parameter(
                    &source.token_emb,
                    [config.vocab_size, config.d_model],
                    device,
                ),
            },
            positions: Embedding {
                weight: parameter(
                    &source.pos_emb,
                    [config.max_seq_len, config.d_model],
                    device,
                ),
            },
            blocks: source
                .blocks
                .iter()
                .map(|block| Block {
                    query: linear(&block.attn.w_q, device),
                    key: linear(&block.attn.w_k, device),
                    value: linear(&block.attn.w_v, device),
                    project: linear(&block.attn.w_o, device),
                    attention_norm: norm(&block.attn.ln, device),
                    expand: linear(&block.ffn.w_1, device),
                    contract: linear(&block.ffn.w_2, device),
                    mlp_norm: norm(&block.ffn.ln, device),
                })
                .collect(),
            norm: norm(&source.ln_f, device),
            head: linear(&source.head, device),
        }
    }

    pub fn forward(&self, tokens: Tensor<2, Int>, heads: usize) -> Tensor<3> {
        let [batch, len] = tokens.dims();
        let device = tokens.device();
        let position_ids = Tensor::<1, Int>::arange(0..len as i64, &device).reshape([1, len]);
        let mut hidden = self.tokens.forward(tokens) + self.positions.forward(position_ids);
        let [_, _, width] = hidden.dims();
        let head_width = width / heads;
        // Burn masks the positions outside the retained lower triangle.
        let mask = Tensor::<2, Bool>::tril_mask([len, len], 0, &device).reshape([1, 1, len, len]);
        let separate_heads = |tensor: Tensor<3>| {
            tensor
                .reshape([batch, len, heads, head_width])
                .swap_dims(1, 2)
        };
        for block in &self.blocks {
            let normalized = block.attention_norm.forward(hidden.clone());
            let query = separate_heads(block.query.forward(normalized.clone()));
            let key = separate_heads(block.key.forward(normalized.clone()));
            let value = separate_heads(block.value.forward(normalized));
            let scores = (query.matmul(key.swap_dims(2, 3)) / (head_width as f32).sqrt())
                .mask_fill(mask.clone(), f32::NEG_INFINITY);
            let context = activation::softmax(scores, 3)
                .matmul(value)
                .swap_dims(1, 2)
                .reshape([batch, len, width]);
            hidden = hidden + block.project.forward(context);
            let expanded = block.expand.forward(block.mlp_norm.forward(hidden.clone()));
            hidden = hidden + block.contract.forward(activation::relu(expanded));
        }
        self.head.forward(self.norm.forward(hidden))
    }

    pub fn parameters(&self) -> Vec<f32> {
        fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
            tensor.to_data_as::<f32>().try_into_vec::<f32>().unwrap()
        }
        fn push_linear(output: &mut Vec<f32>, layer: &Linear) {
            output.extend(values(layer.weight.val().transpose()));
            output.extend(values(layer.bias.as_ref().unwrap().val()));
        }
        fn push_norm(output: &mut Vec<f32>, layer: &LayerNorm) {
            output.extend(values(layer.gamma.val()));
            output.extend(values(layer.beta.as_ref().unwrap().val()));
        }

        let mut output = values(self.tokens.weight.val());
        output.extend(values(self.positions.weight.val()));
        for block in &self.blocks {
            for layer in [&block.query, &block.key, &block.value, &block.project] {
                push_linear(&mut output, layer);
            }
            push_norm(&mut output, &block.attention_norm);
            push_linear(&mut output, &block.expand);
            push_linear(&mut output, &block.contract);
            push_norm(&mut output, &block.mlp_norm);
        }
        push_norm(&mut output, &self.norm);
        push_linear(&mut output, &self.head);
        output
    }
}

#[cfg(all(test, feature = "cycle5-forward-test"))]
mod tests {
    use super::*;
    use moonshot_model::ModelConfig;

    #[test]
    fn matches_native_model_and_parameter_layout() {
        let native = TinyTransformer::new(ModelConfig::new_1m(), 42);
        let device = Device::ndarray();
        let model = Model::from_moonshot(&native, &device);
        assert_eq!(model.num_params(), 1_095_744);
        assert_eq!(model.parameters(), native.extract_parameters());
        let tokens = [[1i64, 26, 32, 33, 3], [1, 26, 35, 41, 3]];
        let actual = model
            .forward(Tensor::from_data(tokens, &device), 8)
            .to_data_as::<f32>()
            .try_into_vec::<f32>()
            .unwrap();
        let expected = tokens
            .iter()
            .flat_map(|row| native.forward(&row.map(|token| token as usize)));
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.005, "{actual} vs {expected}");
        }
    }
}
