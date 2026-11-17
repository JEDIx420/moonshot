//! Safetensors Model Exporter and Importer
//!
//! Standard HuggingFace-compatible binary serialization for Moonshot models (77K and 1.09M).
//! Provides 8-byte memory alignment, UTF-8 JSON header with space padding (`N % 8 == 0`),
//! contiguous little-endian float buffers, and Netron / PyTorch hierarchical tensor naming.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{ModelConfig, TinyTransformer};

/// Errors encountered during Safetensors parsing, serialization, or model loading.
#[derive(Debug, Clone, PartialEq)]
pub enum SafetensorsError {
    /// Input byte slice is smaller than the minimum 8-byte header prefix.
    FileTooSmall { size: usize },
    /// Declared header length exceeds available buffer or sanity threshold.
    HeaderTooLarge { header_len: usize, max_allowed: usize },
    /// Header bytes could not be decoded as valid UTF-8.
    InvalidUtf8(String),
    /// JSON header parsing failed or is structurally invalid.
    InvalidJson(String),
    /// Tensor offsets are invalid (out of bounds, inverted, or exceeding buffer).
    InvalidOffsets {
        tensor: String,
        start: usize,
        end: usize,
        buffer_len: usize,
    },
    /// Element data type is unsupported (only "F32" is supported).
    UnsupportedDtype {
        tensor: String,
        dtype: String,
    },
    /// Tensor dimensions do not match the expected model configuration.
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A required model tensor is absent from the file.
    MissingTensor(String),
    /// Declared byte slice length does not equal `shape.product() * sizeof(dtype)`.
    ByteLengthMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    /// Essential model configuration metadata is absent.
    MetadataMissing(String),
    /// Generic corruption error with diagnostic context.
    CorruptedFile(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileTooSmall { size } => {
                write!(f, "Safetensors file too small ({} bytes, minimum 8 bytes required)", size)
            }
            Self::HeaderTooLarge { header_len, max_allowed } => {
                write!(
                    f,
                    "Safetensors header length {} exceeds maximum allowed {}",
                    header_len, max_allowed
                )
            }
            Self::InvalidUtf8(err) => write!(f, "Invalid UTF-8 in Safetensors header: {}", err),
            Self::InvalidJson(err) => write!(f, "Invalid JSON in Safetensors header: {}", err),
            Self::InvalidOffsets { tensor, start, end, buffer_len } => {
                write!(
                    f,
                    "Invalid offsets for tensor '{}': [{}, {}] in buffer of length {}",
                    tensor, start, end, buffer_len
                )
            }
            Self::UnsupportedDtype { tensor, dtype } => {
                write!(
                    f,
                    "Unsupported dtype '{}' for tensor '{}' (expected 'F32')",
                    dtype, tensor
                )
            }
            Self::ShapeMismatch { tensor, expected, actual } => {
                write!(
                    f,
                    "Shape mismatch for tensor '{}': expected {:?}, found {:?}",
                    tensor, expected, actual
                )
            }
            Self::MissingTensor(tensor) => write!(f, "Required tensor '{}' is missing", tensor),
            Self::ByteLengthMismatch { tensor, expected, actual } => {
                write!(
                    f,
                    "Byte length mismatch for tensor '{}': expected {} bytes, found {}",
                    tensor, expected, actual
                )
            }
            Self::MetadataMissing(field) => {
                write!(f, "Required model metadata field '{}' is missing", field)
            }
            Self::CorruptedFile(msg) => write!(f, "Corrupted Safetensors file: {}", msg),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// View describing a tensor's metadata and location within the binary buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorView {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

impl TensorView {
    pub fn new(
        name: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    ) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            shape,
            data_offsets,
        }
    }

    /// Number of elements in the tensor (product of shape dimensions).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte length occupied in the data buffer.
    pub fn byte_len(&self) -> usize {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }
}

/// Internal struct for deserializing tensor entries from the JSON header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct HeaderEntryJson {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Parsed Safetensors container providing zero-copy metadata inspection and float extraction.
#[derive(Debug, Clone)]
pub struct Safetensors<'a> {
    pub metadata: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, TensorView>,
    pub buffer: &'a [u8],
}

impl<'a> Safetensors<'a> {
    /// Parse and validate a Safetensors binary file from bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, SafetensorsError> {
        if bytes.len() < 8 {
            return Err(SafetensorsError::FileTooSmall { size: bytes.len() });
        }

        let header_len = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| SafetensorsError::FileTooSmall { size: bytes.len() })?,
        ) as usize;

        // Sanity limit: 100 MB max header size
        const MAX_HEADER_LEN: usize = 100 * 1024 * 1024;
        let available_for_header = bytes.len().saturating_sub(8);
        if header_len > MAX_HEADER_LEN || header_len > available_for_header {
            return Err(SafetensorsError::HeaderTooLarge {
                header_len,
                max_allowed: available_for_header.min(MAX_HEADER_LEN),
            });
        }

        let header_slice = &bytes[8..8 + header_len];
        let header_str = std::str::from_utf8(header_slice)
            .map_err(|e| SafetensorsError::InvalidUtf8(e.to_string()))?;

        if !header_str.trim_start().starts_with('{') {
            return Err(SafetensorsError::InvalidJson(
                "Header does not begin with '{'".to_string(),
            ));
        }

        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| SafetensorsError::InvalidJson(e.to_string()))?;

        let data_buffer = &bytes[8 + header_len..];
        let buffer_len = data_buffer.len();

        let mut metadata = BTreeMap::new();
        let mut tensors = BTreeMap::new();

        for (key, val) in raw_map {
            if key == "__metadata__" {
                if let Some(obj) = val.as_object() {
                    for (m_key, m_val) in obj {
                        let m_str = match m_val {
                            serde_json::Value::String(s) => s.clone(),
                            _ => m_val.to_string(),
                        };
                        metadata.insert(m_key.clone(), m_str);
                    }
                }
            } else {
                let entry: HeaderEntryJson = serde_json::from_value(val)
                    .map_err(|e| SafetensorsError::InvalidJson(format!("Tensor '{}': {}", key, e)))?;

                let [start, end] = entry.data_offsets;
                if start > end || end > buffer_len {
                    return Err(SafetensorsError::InvalidOffsets {
                        tensor: key,
                        start,
                        end,
                        buffer_len,
                    });
                }

                if entry.dtype != "F32" {
                    return Err(SafetensorsError::UnsupportedDtype {
                        tensor: key,
                        dtype: entry.dtype,
                    });
                }

                let num_elements: usize = entry.shape.iter().product();
                let expected_bytes = num_elements * 4;
                let actual_bytes = end - start;
                if expected_bytes != actual_bytes {
                    return Err(SafetensorsError::ByteLengthMismatch {
                        tensor: key,
                        expected: expected_bytes,
                        actual: actual_bytes,
                    });
                }

                let view = TensorView {
                    name: key.clone(),
                    dtype: entry.dtype,
                    shape: entry.shape,
                    data_offsets: entry.data_offsets,
                };
                tensors.insert(key, view);
            }
        }

        Ok(Self {
            metadata,
            tensors,
            buffer: data_buffer,
        })
    }

    /// Retrieve metadata value for a given key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Parse an unsigned integer metadata value.
    pub fn get_metadata_usize(&self, key: &str) -> Result<usize, SafetensorsError> {
        self.metadata
            .get(key)
            .ok_or_else(|| SafetensorsError::MetadataMissing(key.to_string()))?
            .parse::<usize>()
            .map_err(|_| {
                SafetensorsError::CorruptedFile(format!("Metadata '{}' is not a valid integer", key))
            })
    }

    /// Look up a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorView> {
        self.tensors.get(name)
    }

    /// List all tensor names in alphabetical order.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Extract float values for a given tensor.
    pub fn get_tensor_data(&self, name: &str) -> Result<Vec<f32>, SafetensorsError> {
        let view = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        let slice = &self.buffer[view.data_offsets[0]..view.data_offsets[1]];
        let mut out = Vec::with_capacity(view.num_elements());
        for chunk in slice.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    /// Copy decoded float values directly into a destination slice.
    pub fn copy_tensor_data(&self, name: &str, dest: &mut [f32]) -> Result<(), SafetensorsError> {
        let view = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        if dest.len() != view.num_elements() {
            return Err(SafetensorsError::ByteLengthMismatch {
                tensor: name.to_string(),
                expected: dest.len() * 4,
                actual: view.byte_len(),
            });
        }
        let slice = &self.buffer[view.data_offsets[0]..view.data_offsets[1]];
        for (chunk, out) in slice.chunks_exact(4).zip(dest.iter_mut()) {
            *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    /// Raw byte slice for a tensor.
    pub fn get_tensor_raw_bytes(&self, name: &str) -> Result<&'a [u8], SafetensorsError> {
        let view = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        Ok(&self.buffer[view.data_offsets[0]..view.data_offsets[1]])
    }
}

/// Serialize arbitrary named tensors into HuggingFace Safetensors format with 8-byte alignment.
pub fn serialize_safetensors(
    tensors: &[(&str, &[usize], &[f32])],
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<Vec<u8>, SafetensorsError> {
    let mut raw_buffer = Vec::new();
    let mut header_map = serde_json::Map::new();

    if let Some(meta) = metadata {
        let meta_val = serde_json::to_value(meta)
            .map_err(|e| SafetensorsError::InvalidJson(e.to_string()))?;
        header_map.insert("__metadata__".to_string(), meta_val);
    }

    for (name, shape, data) in tensors {
        let start = raw_buffer.len();
        let byte_len = data.len() * 4;
        let end = start + byte_len;

        raw_buffer.reserve(byte_len);
        for &val in *data {
            raw_buffer.extend_from_slice(&val.to_le_bytes());
        }

        let entry = serde_json::json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, end],
        });
        header_map.insert((*name).to_string(), entry);
    }

    let json_str = serde_json::to_string(&header_map)
        .map_err(|e| SafetensorsError::InvalidJson(e.to_string()))?;
    let mut json_bytes = json_str.into_bytes();

    // Pad with ASCII space characters (0x20) so that header length N % 8 == 0
    let rem = json_bytes.len() % 8;
    if rem != 0 {
        let pad = 8 - rem;
        json_bytes.resize(json_bytes.len() + pad, b' ');
    }

    let header_len = json_bytes.len() as u64;
    let mut output = Vec::with_capacity(8 + json_bytes.len() + raw_buffer.len());
    output.extend_from_slice(&header_len.to_le_bytes());
    output.extend_from_slice(&json_bytes);
    output.extend_from_slice(&raw_buffer);

    Ok(output)
}

/// Export a `TinyTransformer` to Safetensors format.
///
/// Encodes all 38 canonical tensors under Netron/PyTorch naming hierarchy,
/// adds model architecture metadata, and aligns buffer to 8-byte boundaries.
pub fn export_model(model: &TinyTransformer) -> Result<Vec<u8>, SafetensorsError> {
    let mut metadata = BTreeMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    metadata.insert("architecture".to_string(), "TinyTransformer".to_string());
    metadata.insert("vocab_size".to_string(), model.config.vocab_size.to_string());
    metadata.insert("max_seq_len".to_string(), model.config.max_seq_len.to_string());
    metadata.insert("d_model".to_string(), model.config.d_model.to_string());
    metadata.insert("n_heads".to_string(), model.config.n_heads.to_string());
    metadata.insert("n_layers".to_string(), model.config.n_layers.to_string());
    metadata.insert("d_ff".to_string(), model.config.d_ff.to_string());
    metadata.insert("total_parameters".to_string(), model.total_parameters().to_string());
    metadata.insert("checksum".to_string(), model.checksum());
    metadata.insert("opt_step".to_string(), model.opt_step.to_string());

    // Build tensor descriptor table
    let mut tensor_list: Vec<(String, Vec<usize>, &[f32])> = Vec::with_capacity(38);

    let d = model.config.d_model;
    let v = model.config.vocab_size;
    let s = model.config.max_seq_len;
    let d_ff = model.config.d_ff;

    // 1. Embeddings
    tensor_list.push(("token_emb.weight".to_string(), vec![v, d], &model.token_emb));
    tensor_list.push(("pos_emb.weight".to_string(), vec![s, d], &model.pos_emb));

    // 2. Transformer Blocks (16 tensors per block)
    for (l, b) in model.blocks.iter().enumerate() {
        tensor_list.push((format!("blocks.{l}.attn.w_q.weight"), vec![d, d], &b.attn.w_q.weight));
        tensor_list.push((format!("blocks.{l}.attn.w_q.bias"), vec![d], &b.attn.w_q.bias));
        tensor_list.push((format!("blocks.{l}.attn.w_k.weight"), vec![d, d], &b.attn.w_k.weight));
        tensor_list.push((format!("blocks.{l}.attn.w_k.bias"), vec![d], &b.attn.w_k.bias));
        tensor_list.push((format!("blocks.{l}.attn.w_v.weight"), vec![d, d], &b.attn.w_v.weight));
        tensor_list.push((format!("blocks.{l}.attn.w_v.bias"), vec![d], &b.attn.w_v.bias));
        tensor_list.push((format!("blocks.{l}.attn.w_o.weight"), vec![d, d], &b.attn.w_o.weight));
        tensor_list.push((format!("blocks.{l}.attn.w_o.bias"), vec![d], &b.attn.w_o.bias));

        // LayerNorm: Netron/PyTorch expects .weight and .bias
        tensor_list.push((format!("blocks.{l}.attn.ln.weight"), vec![d], &b.attn.ln.gamma));
        tensor_list.push((format!("blocks.{l}.attn.ln.bias"), vec![d], &b.attn.ln.beta));

        tensor_list.push((format!("blocks.{l}.ffn.w_1.weight"), vec![d_ff, d], &b.ffn.w_1.weight));
        tensor_list.push((format!("blocks.{l}.ffn.w_1.bias"), vec![d_ff], &b.ffn.w_1.bias));
        tensor_list.push((format!("blocks.{l}.ffn.w_2.weight"), vec![d, d_ff], &b.ffn.w_2.weight));
        tensor_list.push((format!("blocks.{l}.ffn.w_2.bias"), vec![d], &b.ffn.w_2.bias));

        tensor_list.push((format!("blocks.{l}.ffn.ln.weight"), vec![d], &b.ffn.ln.gamma));
        tensor_list.push((format!("blocks.{l}.ffn.ln.bias"), vec![d], &b.ffn.ln.beta));

        // Optional LoRA adapter
        if let Some(ref lora) = b.attn.adapter {
            tensor_list.push((format!("blocks.{l}.attn.adapter.a_q"), vec![lora.rank, d], &lora.a_q));
            tensor_list.push((format!("blocks.{l}.attn.adapter.b_q"), vec![d, lora.rank], &lora.b_q));
            tensor_list.push((format!("blocks.{l}.attn.adapter.a_v"), vec![lora.rank, d], &lora.a_v));
            tensor_list.push((format!("blocks.{l}.attn.adapter.b_v"), vec![d, lora.rank], &lora.b_v));
        }
    }

    // 3. Final LayerNorm & Language Modeling Head
    tensor_list.push(("ln_f.weight".to_string(), vec![d], &model.ln_f.gamma));
    tensor_list.push(("ln_f.bias".to_string(), vec![d], &model.ln_f.beta));
    tensor_list.push(("head.weight".to_string(), vec![v, d], &model.head.weight));
    tensor_list.push(("head.bias".to_string(), vec![v], &model.head.bias));

    let borrowed_tensors: Vec<(&str, &[usize], &[f32])> = tensor_list
        .iter()
        .map(|(n, s, d)| (n.as_str(), s.as_slice(), *d))
        .collect();

    serialize_safetensors(&borrowed_tensors, Some(&metadata))
}

/// Helper function to load a tensor into a destination slice, trying canonical names and aliases.
fn load_tensor_with_aliases(
    st: &Safetensors,
    canonical_name: &str,
    aliases: &[&str],
    expected_shape: &[usize],
    dest: &mut [f32],
) -> Result<(), SafetensorsError> {
    let found_name = if st.tensors.contains_key(canonical_name) {
        Some(canonical_name)
    } else {
        aliases.iter().find(|&&a| st.tensors.contains_key(a)).copied()
    };

    let name = match found_name {
        Some(n) => n,
        None => return Err(SafetensorsError::MissingTensor(canonical_name.to_string())),
    };

    let view = &st.tensors[name];
    if view.shape != expected_shape {
        return Err(SafetensorsError::ShapeMismatch {
            tensor: canonical_name.to_string(),
            expected: expected_shape.to_vec(),
            actual: view.shape.clone(),
        });
    }

    if dest.len() != view.num_elements() {
        return Err(SafetensorsError::ByteLengthMismatch {
            tensor: canonical_name.to_string(),
            expected: dest.len() * 4,
            actual: view.byte_len(),
        });
    }

    st.copy_tensor_data(name, dest)?;
    Ok(())
}

/// Load Safetensors parameters directly into an existing `TinyTransformer` instance.
pub fn load_into_model(model: &mut TinyTransformer, bytes: &[u8]) -> Result<(), SafetensorsError> {
    let st = Safetensors::parse(bytes)?;

    let d = model.config.d_model;
    let v = model.config.vocab_size;
    let s = model.config.max_seq_len;
    let d_ff = model.config.d_ff;

    // 1. Embeddings
    load_tensor_with_aliases(
        &st,
        "token_emb.weight",
        &["token_emb"],
        &[v, d],
        &mut model.token_emb,
    )?;
    load_tensor_with_aliases(
        &st,
        "pos_emb.weight",
        &["pos_emb"],
        &[s, d],
        &mut model.pos_emb,
    )?;

    // 2. Transformer Blocks
    for (l, b) in model.blocks.iter_mut().enumerate() {
        let prefix = format!("blocks.{l}");

        let w_q_w = format!("{prefix}.attn.w_q.weight");
        load_tensor_with_aliases(&st, &w_q_w, &[], &[d, d], &mut b.attn.w_q.weight)?;
        let w_q_b = format!("{prefix}.attn.w_q.bias");
        load_tensor_with_aliases(&st, &w_q_b, &[], &[d], &mut b.attn.w_q.bias)?;

        let w_k_w = format!("{prefix}.attn.w_k.weight");
        load_tensor_with_aliases(&st, &w_k_w, &[], &[d, d], &mut b.attn.w_k.weight)?;
        let w_k_b = format!("{prefix}.attn.w_k.bias");
        load_tensor_with_aliases(&st, &w_k_b, &[], &[d], &mut b.attn.w_k.bias)?;

        let w_v_w = format!("{prefix}.attn.w_v.weight");
        load_tensor_with_aliases(&st, &w_v_w, &[], &[d, d], &mut b.attn.w_v.weight)?;
        let w_v_b = format!("{prefix}.attn.w_v.bias");
        load_tensor_with_aliases(&st, &w_v_b, &[], &[d], &mut b.attn.w_v.bias)?;

        let w_o_w = format!("{prefix}.attn.w_o.weight");
        load_tensor_with_aliases(&st, &w_o_w, &[], &[d, d], &mut b.attn.w_o.weight)?;
        let w_o_b = format!("{prefix}.attn.w_o.bias");
        load_tensor_with_aliases(&st, &w_o_b, &[], &[d], &mut b.attn.w_o.bias)?;

        let ln_w = format!("{prefix}.attn.ln.weight");
        let ln_gamma_alias = format!("{prefix}.attn.ln.gamma");
        load_tensor_with_aliases(
            &st,
            &ln_w,
            &[&ln_gamma_alias],
            &[d],
            &mut b.attn.ln.gamma,
        )?;
        let ln_b = format!("{prefix}.attn.ln.bias");
        let ln_beta_alias = format!("{prefix}.attn.ln.beta");
        load_tensor_with_aliases(
            &st,
            &ln_b,
            &[&ln_beta_alias],
            &[d],
            &mut b.attn.ln.beta,
        )?;

        let ffn_w1_w = format!("{prefix}.ffn.w_1.weight");
        load_tensor_with_aliases(&st, &ffn_w1_w, &[], &[d_ff, d], &mut b.ffn.w_1.weight)?;
        let ffn_w1_b = format!("{prefix}.ffn.w_1.bias");
        load_tensor_with_aliases(&st, &ffn_w1_b, &[], &[d_ff], &mut b.ffn.w_1.bias)?;

        let ffn_w2_w = format!("{prefix}.ffn.w_2.weight");
        load_tensor_with_aliases(&st, &ffn_w2_w, &[], &[d, d_ff], &mut b.ffn.w_2.weight)?;
        let ffn_w2_b = format!("{prefix}.ffn.w_2.bias");
        load_tensor_with_aliases(&st, &ffn_w2_b, &[], &[d], &mut b.ffn.w_2.bias)?;

        let ffn_ln_w = format!("{prefix}.ffn.ln.weight");
        let ffn_ln_gamma_alias = format!("{prefix}.ffn.ln.gamma");
        load_tensor_with_aliases(
            &st,
            &ffn_ln_w,
            &[&ffn_ln_gamma_alias],
            &[d],
            &mut b.ffn.ln.gamma,
        )?;
        let ffn_ln_b = format!("{prefix}.ffn.ln.bias");
        let ffn_ln_beta_alias = format!("{prefix}.ffn.ln.beta");
        load_tensor_with_aliases(
            &st,
            &ffn_ln_b,
            &[&ffn_ln_beta_alias],
            &[d],
            &mut b.ffn.ln.beta,
        )?;

        // Optional adapter loading if present
        if let Some(ref mut lora) = b.attn.adapter {
            let a_q_name = format!("{prefix}.attn.adapter.a_q");
            let b_q_name = format!("{prefix}.attn.adapter.b_q");
            let a_v_name = format!("{prefix}.attn.adapter.a_v");
            let b_v_name = format!("{prefix}.attn.adapter.b_v");

            if st.tensors.contains_key(&a_q_name) {
                let _ = load_tensor_with_aliases(&st, &a_q_name, &[], &[lora.rank, d], &mut lora.a_q);
            }
            if st.tensors.contains_key(&b_q_name) {
                let _ = load_tensor_with_aliases(&st, &b_q_name, &[], &[d, lora.rank], &mut lora.b_q);
            }
            if st.tensors.contains_key(&a_v_name) {
                let _ = load_tensor_with_aliases(&st, &a_v_name, &[], &[lora.rank, d], &mut lora.a_v);
            }
            if st.tensors.contains_key(&b_v_name) {
                let _ = load_tensor_with_aliases(&st, &b_v_name, &[], &[d, lora.rank], &mut lora.b_v);
            }
        }
    }

    // 3. Final LayerNorm & Head
    load_tensor_with_aliases(&st, "ln_f.weight", &["ln_f.gamma"], &[d], &mut model.ln_f.gamma)?;
    load_tensor_with_aliases(&st, "ln_f.bias", &["ln_f.beta"], &[d], &mut model.ln_f.beta)?;
    load_tensor_with_aliases(&st, "head.weight", &[], &[v, d], &mut model.head.weight)?;
    load_tensor_with_aliases(&st, "head.bias", &[], &[v], &mut model.head.bias)?;

    if let Ok(step) = st.get_metadata_usize("opt_step") {
        model.opt_step = step;
    }

    Ok(())
}

/// Infer `ModelConfig` from tensor shapes when metadata is incomplete or absent.
fn infer_config_from_tensors(st: &Safetensors) -> Result<ModelConfig, SafetensorsError> {
    // 1. vocab_size & d_model from token_emb
    let token_emb = st
        .get_tensor("token_emb.weight")
        .or_else(|| st.get_tensor("token_emb"))
        .ok_or_else(|| SafetensorsError::MissingTensor("token_emb.weight".to_string()))?;
    if token_emb.shape.len() != 2 {
        return Err(SafetensorsError::CorruptedFile(
            "token_emb shape must have 2 dimensions".to_string(),
        ));
    }
    let vocab_size = token_emb.shape[0];
    let d_model = token_emb.shape[1];

    // 2. max_seq_len from pos_emb
    let pos_emb = st
        .get_tensor("pos_emb.weight")
        .or_else(|| st.get_tensor("pos_emb"))
        .ok_or_else(|| SafetensorsError::MissingTensor("pos_emb.weight".to_string()))?;
    if pos_emb.shape.len() != 2 {
        return Err(SafetensorsError::CorruptedFile(
            "pos_emb shape must have 2 dimensions".to_string(),
        ));
    }
    let max_seq_len = pos_emb.shape[0];

    // 3. d_ff from blocks.0.ffn.w_1.weight
    let ffn_w1 = st
        .get_tensor("blocks.0.ffn.w_1.weight")
        .ok_or_else(|| SafetensorsError::MissingTensor("blocks.0.ffn.w_1.weight".to_string()))?;
    if ffn_w1.shape.len() != 2 {
        return Err(SafetensorsError::CorruptedFile(
            "blocks.0.ffn.w_1.weight shape must have 2 dimensions".to_string(),
        ));
    }
    let d_ff = ffn_w1.shape[0];

    // 4. n_layers: count unique block indices
    let mut max_block_idx = 0;
    for name in st.tensors.keys() {
        if let Some(rest) = name.strip_prefix("blocks.") {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx >= max_block_idx {
                        max_block_idx = idx + 1;
                    }
                }
            }
        }
    }
    let n_layers = max_block_idx.max(1);

    // 5. n_heads: from metadata if available, otherwise heuristic based on standard configurations
    let n_heads = if let Ok(h) = st.get_metadata_usize("n_heads") {
        h
    } else if d_model == 64 {
        4
    } else if d_model == 256 {
        8
    } else {
        (d_model / 16).max(1)
    };

    Ok(ModelConfig {
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        d_ff,
    })
}

/// Import a full `TinyTransformer` from Safetensors bytes, reconstructing configuration automatically.
pub fn import_model(bytes: &[u8]) -> Result<TinyTransformer, SafetensorsError> {
    let st = Safetensors::parse(bytes)?;

    let config = if let (Ok(v), Ok(s), Ok(d), Ok(h), Ok(l), Ok(f)) = (
        st.get_metadata_usize("vocab_size"),
        st.get_metadata_usize("max_seq_len"),
        st.get_metadata_usize("d_model"),
        st.get_metadata_usize("n_heads"),
        st.get_metadata_usize("n_layers"),
        st.get_metadata_usize("d_ff"),
    ) {
        ModelConfig {
            vocab_size: v,
            max_seq_len: s,
            d_model: d,
            n_heads: h,
            n_layers: l,
            d_ff: f,
        }
    } else {
        infer_config_from_tensors(&st)?
    };

    let mut model = TinyTransformer::new(config, 0);
    load_into_model(&mut model, bytes)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    #[test]
    fn test_safetensors_77k_roundtrip_bitwise() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 42);
        assert_eq!(model.total_parameters(), 77_376);

        let bytes = export_model(&model).expect("export should succeed");
        assert!(bytes.len() > 77_376 * 4);

        let restored = import_model(&bytes).expect("import should succeed");
        assert_eq!(restored.total_parameters(), 77_376);
        assert_eq!(model.config, restored.config);

        // Bitwise parameter comparison across all 77,376 parameters
        let orig_params = model.extract_parameters();
        let rest_params = restored.extract_parameters();
        assert_eq!(orig_params.len(), rest_params.len());
        for (i, (&orig, &rest)) in orig_params.iter().zip(rest_params.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                rest.to_bits(),
                "Parameter mismatch at index {}: {} != {}",
                i,
                orig,
                rest
            );
        }
    }

    #[test]
    fn test_safetensors_1m_roundtrip_bitwise() {
        let model = TinyTransformer::new(ModelConfig::new_1m(), 1337);
        assert_eq!(model.total_parameters(), 1_095_744);

        let bytes = export_model(&model).expect("export 1M model should succeed");
        assert!(bytes.len() > 1_095_744 * 4);

        let restored = import_model(&bytes).expect("import 1M model should succeed");
        assert_eq!(restored.total_parameters(), 1_095_744);
        assert_eq!(model.config, restored.config);

        let orig_params = model.extract_parameters();
        let rest_params = restored.extract_parameters();
        assert_eq!(orig_params.len(), rest_params.len());
        for (i, (&orig, &rest)) in orig_params.iter().zip(rest_params.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                rest.to_bits(),
                "Parameter mismatch at index {}: {} != {}",
                i,
                orig,
                rest
            );
        }
    }

    #[test]
    fn test_safetensors_forward_logits_equality() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 2026);
        let bytes = export_model(&model).expect("export should succeed");
        let restored = import_model(&bytes).expect("import should succeed");

        let tokens = vec![1, 12, 33, 44, 2];
        let orig_logits = model.forward(&tokens);
        let rest_logits = restored.forward(&tokens);

        assert_eq!(orig_logits.len(), rest_logits.len());
        for (i, (&o, &r)) in orig_logits.iter().zip(rest_logits.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                r.to_bits(),
                "Logit mismatch at index {}: {} != {}",
                i,
                o,
                r
            );
        }
    }

    #[test]
    fn test_safetensors_8byte_alignment() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 99);
        let bytes = export_model(&model).expect("export should succeed");

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0, "Header length must be a multiple of 8");
        assert_eq!(
            (8 + header_len) % 8,
            0,
            "Data buffer offset must be 8-byte aligned"
        );
    }

    #[test]
    fn test_safetensors_corrupted_header_handling() {
        // 1. File too small (< 8 bytes)
        let too_small = vec![0u8; 4];
        let err = Safetensors::parse(&too_small).unwrap_err();
        assert!(matches!(err, SafetensorsError::FileTooSmall { size: 4 }));

        // 2. Declared header length exceeds available buffer
        let mut bad_header = Vec::new();
        bad_header.extend_from_slice(&1000u64.to_le_bytes());
        bad_header.extend_from_slice(&[b' '; 10]);
        let err = Safetensors::parse(&bad_header).unwrap_err();
        assert!(matches!(err, SafetensorsError::HeaderTooLarge { .. }));

        // 3. Invalid UTF-8 in header
        let mut bad_utf8 = Vec::new();
        bad_utf8.extend_from_slice(&8u64.to_le_bytes());
        bad_utf8.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8]);
        let err = Safetensors::parse(&bad_utf8).unwrap_err();
        assert!(matches!(err, SafetensorsError::InvalidUtf8(_)));

        // 4. Invalid JSON (not starting with '{')
        let mut not_json = Vec::new();
        let payload = b"hello world 123 ";
        not_json.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        not_json.extend_from_slice(payload);
        let err = Safetensors::parse(&not_json).unwrap_err();
        assert!(matches!(err, SafetensorsError::InvalidJson(_)));

        // 5. Malformed JSON syntax
        let mut malformed_json = Vec::new();
        let payload = b"{ \"a\": ";
        malformed_json.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        malformed_json.extend_from_slice(payload);
        let err = Safetensors::parse(&malformed_json).unwrap_err();
        assert!(matches!(err, SafetensorsError::InvalidJson(_)));
    }

    #[test]
    fn test_safetensors_shape_mismatch_error() {
        // Export 1M model (d_model=256)
        let model_1m = TinyTransformer::new(ModelConfig::new_1m(), 7);
        let bytes_1m = export_model(&model_1m).expect("export should succeed");

        // Attempt to load into 77K model (d_model=64)
        let mut model_77k = TinyTransformer::new(ModelConfig::new_77k(), 7);
        let err = load_into_model(&mut model_77k, &bytes_1m).unwrap_err();
        assert!(
            matches!(err, SafetensorsError::ShapeMismatch { .. }),
            "Expected ShapeMismatch, got {:?}",
            err
        );
    }

    #[test]
    fn test_safetensors_missing_tensor_error() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 42);
        // Serialize missing "head.weight"
        let tensors = [
            ("token_emb.weight", &[64, 64][..], &model.token_emb[..]),
        ];
        let bytes = serialize_safetensors(&tensors, None).expect("serialization should succeed");

        let mut target = TinyTransformer::new(ModelConfig::new_77k(), 42);
        let err = load_into_model(&mut target, &bytes).unwrap_err();
        assert!(
            matches!(err, SafetensorsError::MissingTensor(_)),
            "Expected MissingTensor, got {:?}",
            err
        );
    }

    #[test]
    fn test_safetensors_alias_compatibility() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 123);
        let bytes = export_model(&model).expect("export should succeed");

        // Parse and replace .weight/.bias with .gamma/.beta for LayerNorm in JSON header
        let st = Safetensors::parse(&bytes).expect("parse should succeed");
        assert!(st.tensors.contains_key("ln_f.weight"));
        assert!(st.tensors.contains_key("blocks.0.attn.ln.weight"));

        // Build a modified header using aliases
        let mut aliased_tensors = Vec::new();
        for (name, view) in &st.tensors {
            let aliased_name = if name == "ln_f.weight" {
                "ln_f.gamma".to_string()
            } else if name == "ln_f.bias" {
                "ln_f.beta".to_string()
            } else if name.ends_with(".attn.ln.weight") {
                name.replace(".attn.ln.weight", ".attn.ln.gamma")
            } else if name.ends_with(".attn.ln.bias") {
                name.replace(".attn.ln.bias", ".attn.ln.beta")
            } else {
                name.clone()
            };
            let data = st.get_tensor_data(name).unwrap();
            aliased_tensors.push((aliased_name, view.shape.clone(), data));
        }

        let tensor_refs: Vec<(&str, &[usize], &[f32])> = aliased_tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();

        let aliased_bytes =
            serialize_safetensors(&tensor_refs, Some(&st.metadata)).expect("serialize should succeed");

        // Load into fresh model and verify bitwise equality
        let mut restored = TinyTransformer::new(ModelConfig::new_77k(), 0);
        load_into_model(&mut restored, &aliased_bytes).expect("alias loading should succeed");

        let orig_params = model.extract_parameters();
        let rest_params = restored.extract_parameters();
        assert_eq!(orig_params, rest_params);
    }

    #[test]
    fn test_safetensors_reconstruct_without_metadata() {
        let model = TinyTransformer::new(ModelConfig::new_77k(), 314);
        let bytes = export_model(&model).expect("export should succeed");

        // Parse and strip __metadata__
        let st = Safetensors::parse(&bytes).expect("parse should succeed");
        let tensor_views: Vec<(&str, &[usize], Vec<f32>)> = st
            .tensors
            .iter()
            .map(|(n, v)| (n.as_str(), v.shape.as_slice(), st.get_tensor_data(n).unwrap()))
            .collect();

        let tensor_refs: Vec<(&str, &[usize], &[f32])> = tensor_views
            .iter()
            .map(|(n, s, d)| (*n, *s, d.as_slice()))
            .collect();

        let bytes_no_meta =
            serialize_safetensors(&tensor_refs, None).expect("serialize without metadata");

        // Reconstruct model: will use infer_config_from_tensors!
        let restored = import_model(&bytes_no_meta).expect("import without metadata should succeed");
        assert_eq!(restored.config, model.config);
        assert_eq!(model.extract_parameters(), restored.extract_parameters());
    }

    #[test]
    fn test_safetensors_export_to_target_for_external_validation() {
        let m77k = TinyTransformer::new(ModelConfig::new_77k(), 42);
        let b77k = export_model(&m77k).expect("export 77k");
        std::fs::create_dir_all("target").ok();
        let _ = std::fs::write("target/test_77k.safetensors", &b77k);
        std::fs::create_dir_all("../../target").ok();
        let _ = std::fs::write("../../target/test_77k.safetensors", &b77k);

        let m1m = TinyTransformer::new(ModelConfig::new_1m(), 42);
        let b1m = export_model(&m1m).expect("export 1m");
        let _ = std::fs::write("target/test_1m.safetensors", &b1m);
        let _ = std::fs::write("../../target/test_1m.safetensors", &b1m);
    }
}
