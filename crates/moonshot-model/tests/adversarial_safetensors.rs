//! Adversarial Stress Testing Suite for Safetensors Exporter and Importer
//! Verifies:
//! 1. Strict 8-byte memory alignment (header length and buffer offset % 8 == 0)
//! 2. Complete 38-tensor bitwise roundtrip for 77K and 1.09M models
//! 3. Zero-panic error handling on corrupted, truncated, fuzzed, and malicious byte buffers
//! 4. Rigorous validation of dtypes, offsets, and tensor shape consistency

use moonshot_model::safetensors::{
    export_model, import_model, load_into_model, serialize_safetensors, Safetensors,
    SafetensorsError,
};
use moonshot_model::{ModelConfig, TinyTransformer};

#[test]
fn test_strict_8byte_alignment_77k_and_1m() {
    for (name, config) in [
        ("77K", ModelConfig::new_77k()),
        ("1.09M", ModelConfig::new_1m()),
    ] {
        let model = TinyTransformer::new(config, 42);
        let bytes = export_model(&model).unwrap();

        assert!(
            bytes.len() >= 8,
            "[{name}] Exported bytes too small: {}",
            bytes.len()
        );

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!(
            header_len % 8,
            0,
            "[{name}] Header length {header_len} is NOT a multiple of 8"
        );

        let buffer_offset = 8 + header_len;
        assert_eq!(
            buffer_offset % 8,
            0,
            "[{name}] Buffer offset {buffer_offset} is NOT 8-byte aligned"
        );

        // Verify each tensor data offset is 4-byte aligned (F32)
        let st = Safetensors::parse(&bytes).unwrap();
        assert_eq!(st.tensors.len(), 38, "[{name}] Must have exactly 38 tensors");
        for (tname, view) in &st.tensors {
            assert_eq!(
                view.data_offsets[0] % 4,
                0,
                "[{name}] Tensor {tname} start offset not 4-byte aligned: {}",
                view.data_offsets[0]
            );
            assert_eq!(
                view.data_offsets[1] % 4,
                0,
                "[{name}] Tensor {tname} end offset not 4-byte aligned: {}",
                view.data_offsets[1]
            );
        }
    }
}

#[test]
fn test_roundtrip_bitwise_fidelity_and_logits_77k_and_1m() {
    let test_tokens = [
        vec![1, 5, 12, 23, 2],
        vec![42, 63, 0, 15, 30],
        vec![7, 7, 7, 7],
    ];

    for (name, config) in [
        ("77K", ModelConfig::new_77k()),
        ("1.09M", ModelConfig::new_1m()),
    ] {
        let model = TinyTransformer::new(config, 12345);
        let bytes = export_model(&model).unwrap();

        let restored = import_model(&bytes).unwrap();
        assert_eq!(model.config, restored.config, "[{name}] Config mismatch");
        assert_eq!(
            model.total_parameters(),
            restored.total_parameters(),
            "[{name}] Parameter count mismatch"
        );

        // Bitwise parameter equality across all weights
        let orig_params = model.extract_parameters();
        let rest_params = restored.extract_parameters();
        assert_eq!(orig_params.len(), rest_params.len());
        for (i, (&o, &r)) in orig_params.iter().zip(rest_params.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                r.to_bits(),
                "[{name}] Bitwise parameter mismatch at index {i}: {o} != {r}"
            );
        }

        // Bitwise forward logits equality across diverse token sequences
        for tokens in &test_tokens {
            let orig_logits = model.forward(tokens);
            let rest_logits = restored.forward(tokens);
            assert_eq!(orig_logits.len(), rest_logits.len());
            for (i, (&o, &r)) in orig_logits.iter().zip(rest_logits.iter()).enumerate() {
                assert_eq!(
                    o.to_bits(),
                    r.to_bits(),
                    "[{name}] Logit mismatch for tokens {tokens:?} at pos {i}: {o} != {r}"
                );
            }
        }
    }
}

#[test]
fn test_corrupted_buffer_handling_no_panics() {
    // 1. Empty buffer
    assert!(matches!(
        Safetensors::parse(&[]),
        Err(SafetensorsError::FileTooSmall { size: 0 })
    ));

    // 2. 1 to 7 bytes
    for size in 1..8 {
        let buf = vec![0u8; size];
        assert!(matches!(
            Safetensors::parse(&buf),
            Err(SafetensorsError::FileTooSmall { .. })
        ));
    }

    // 3. 8-byte zeros buffer -> header_len = 0
    let zeros_8 = [0u8; 8];
    assert!(matches!(
        Safetensors::parse(&zeros_8),
        Err(SafetensorsError::InvalidJson(_))
    ));

    // 4. Declared header length exceeds available buffer
    let mut bad_len = Vec::new();
    bad_len.extend_from_slice(&1024u64.to_le_bytes());
    bad_len.extend_from_slice(b"{}");
    assert!(matches!(
        Safetensors::parse(&bad_len),
        Err(SafetensorsError::HeaderTooLarge { .. })
    ));

    // 5. Huge declared header length (u64::MAX)
    let mut max_len = Vec::new();
    max_len.extend_from_slice(&u64::MAX.to_le_bytes());
    max_len.extend_from_slice(b"{}");
    assert!(matches!(
        Safetensors::parse(&max_len),
        Err(SafetensorsError::HeaderTooLarge { .. })
    ));

    // 6. Header bytes are invalid UTF-8
    let mut bad_utf8 = Vec::new();
    bad_utf8.extend_from_slice(&8u64.to_le_bytes());
    bad_utf8.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8]);
    assert!(matches!(
        Safetensors::parse(&bad_utf8),
        Err(SafetensorsError::InvalidUtf8(_))
    ));

    // 7. Header starts with something other than '{'
    let not_json_header = b"not_json_header!";
    let mut not_json = Vec::new();
    not_json.extend_from_slice(&(not_json_header.len() as u64).to_le_bytes());
    not_json.extend_from_slice(not_json_header);
    assert!(matches!(
        Safetensors::parse(&not_json),
        Err(SafetensorsError::InvalidJson(_))
    ));

    // 8. Header is JSON array instead of object
    let json_array = b"[]      ";
    let mut arr_buf = Vec::new();
    arr_buf.extend_from_slice(&(json_array.len() as u64).to_le_bytes());
    arr_buf.extend_from_slice(json_array);
    assert!(matches!(
        Safetensors::parse(&arr_buf),
        Err(SafetensorsError::InvalidJson(_))
    ));
}

#[test]
fn test_corrupted_tensor_metadata_rejection() {
    // Helper to build a safetensors binary from raw JSON string and raw buffer
    let build_raw = |json_str: &str, raw_data: &[u8]| -> Vec<u8> {
        let mut json_bytes = json_str.as_bytes().to_vec();
        let rem = json_bytes.len() % 8;
        if rem != 0 {
            json_bytes.resize(json_bytes.len() + (8 - rem), b' ');
        }
        let header_len = json_bytes.len() as u64;
        let mut out = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(raw_data);
        out
    };

    // 1. Inverted offsets (start > end)
    let bad_offsets = r#"{"test": {"dtype": "F32", "shape": [2], "data_offsets": [8, 4]}}"#;
    let bytes = build_raw(bad_offsets, &[0u8; 16]);
    assert!(matches!(
        Safetensors::parse(&bytes),
        Err(SafetensorsError::InvalidOffsets { .. })
    ));

    // 2. Offsets out of buffer bounds (end > buffer.len())
    let oob_offsets = r#"{"test": {"dtype": "F32", "shape": [2], "data_offsets": [0, 64]}}"#;
    let bytes = build_raw(oob_offsets, &[0u8; 16]); // only 16 bytes available
    assert!(matches!(
        Safetensors::parse(&bytes),
        Err(SafetensorsError::InvalidOffsets { .. })
    ));

    // 3. Unsupported dtype ("F16")
    let f16_dtype = r#"{"test": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}"#;
    let bytes = build_raw(f16_dtype, &[0u8; 8]);
    assert!(matches!(
        Safetensors::parse(&bytes),
        Err(SafetensorsError::UnsupportedDtype { .. })
    ));

    // 4. Byte length mismatch: shape [3] requires 12 bytes, but offsets are [0, 8] (8 bytes)
    let mismatch = r#"{"test": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}"#;
    let bytes = build_raw(mismatch, &[0u8; 16]);
    assert!(matches!(
        Safetensors::parse(&bytes),
        Err(SafetensorsError::ByteLengthMismatch { .. })
    ));
}

#[test]
fn test_fuzz_random_buffers_no_panics() {
    // Generate pseudo-random noise and ensure parse NEVER panics
    let mut seed: u64 = 0xDEADBEEF;
    let lcg = |s: &mut u64| -> u8 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s >> 33) as u8
    };

    for len in [0, 1, 7, 8, 9, 15, 16, 32, 64, 128, 512, 2048] {
        let mut noise = Vec::with_capacity(len);
        for _ in 0..len {
            noise.push(lcg(&mut seed));
        }
        let _ = Safetensors::parse(&noise); // Must not panic!
    }
}

#[test]
fn test_load_into_model_safety_checks() {
    let mut model_77k = TinyTransformer::new(ModelConfig::new_77k(), 42);

    // 1. Missing required tensor
    let sparse_tensor = [("token_emb.weight", &[64, 64][..], &model_77k.token_emb[..])];
    let sparse_bytes = serialize_safetensors(&sparse_tensor, None).unwrap();
    let err = load_into_model(&mut model_77k, &sparse_bytes).unwrap_err();
    assert!(matches!(err, SafetensorsError::MissingTensor(_)));

    // 2. Shape mismatch (trying to load 1.09M weights into 77K model)
    let model_1m = TinyTransformer::new(ModelConfig::new_1m(), 42);
    let bytes_1m = export_model(&model_1m).unwrap();
    let err = load_into_model(&mut model_77k, &bytes_1m).unwrap_err();
    assert!(matches!(err, SafetensorsError::ShapeMismatch { .. }));
}
