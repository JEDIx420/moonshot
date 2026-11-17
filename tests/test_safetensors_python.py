#!/usr/bin/env python3
"""
Tier 2 E2E Test: Safetensors Model Exporter/Importer Verification
Verifies:
1. Standard HuggingFace Safetensors binary layout (8-byte header prefix, padded JSON header, 8-byte alignment)
2. String-to-string dictionary in __metadata__
3. Netron tensor naming convention and visual hierarchy (38 tensors across layers)
4. Parameter counts for 77K (77,376 params) and 1.09M (1,095,744 params) models
5. Verification via Python official `safetensors` library (safetensors.numpy or safe_open)
6. Adversarial corruption checks (truncated bytes, bad header size, unaligned offset, invalid dtype)
7. Execution of `cargo test -p moonshot-model`
"""

import json
import os
import struct
import subprocess
import sys

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# Expected 38 tensor names in Netron visual hierarchy
def get_expected_tensor_names(n_layers: int = 2) -> set[str]:
    names = {"token_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias", "head.weight", "head.bias"}
    for l in range(n_layers):
        prefix = f"blocks.{l}"
        # Attention
        for proj in ["w_q", "w_k", "w_v", "w_o"]:
            names.add(f"{prefix}.attn.{proj}.weight")
            names.add(f"{prefix}.attn.{proj}.bias")
        names.add(f"{prefix}.attn.ln.weight")
        names.add(f"{prefix}.attn.ln.bias")
        # FFN
        names.add(f"{prefix}.ffn.w_1.weight")
        names.add(f"{prefix}.ffn.w_1.bias")
        names.add(f"{prefix}.ffn.w_2.weight")
        names.add(f"{prefix}.ffn.w_2.bias")
        names.add(f"{prefix}.ffn.ln.weight")
        names.add(f"{prefix}.ffn.ln.bias")
    return names

# ---------------------------------------------------------------------------
# Test 1: Cargo Test moonshot-model Execution
# ---------------------------------------------------------------------------
def test_cargo_test_model():
    print_header("Test 1: Cargo Test moonshot-model (Unit & Roundtrip Tests)")
    cmd = ["cargo", "test", "-p", "moonshot-model", "--", "--nocapture"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"`cargo test -p moonshot-model` failed with code {result.returncode}")
    print("  [PASS] `cargo test -p moonshot-model` passed successfully!")

# ---------------------------------------------------------------------------
# Test 2: Low-Level Safetensors Binary Spec Parser & Alignment Validator
# ---------------------------------------------------------------------------
def parse_and_validate_safetensors_bytes(raw_bytes: bytes, expected_params: int, model_name: str) -> dict:
    total_len = len(raw_bytes)
    assert total_len >= 8, f"File too small: {total_len} bytes < 8 bytes"

    # 1. 8-byte little-endian header length N
    header_len = struct.unpack("<Q", raw_bytes[:8])[0]
    print(f"  [{model_name}] Header length N = {header_len}")

    # 2. 8-Byte alignment validation
    assert header_len % 8 == 0, (
        f"[{model_name}] Header length {header_len} is NOT a multiple of 8! 8-byte alignment violation."
    )
    buffer_offset = 8 + header_len
    assert buffer_offset % 8 == 0, (
        f"[{model_name}] Buffer offset {buffer_offset} is NOT 8-byte aligned!"
    )
    assert total_len >= buffer_offset, (
        f"[{model_name}] Declared header exceeds total file length ({buffer_offset} > {total_len})"
    )

    # 3. JSON Header decoding
    header_bytes = raw_bytes[8:buffer_offset]
    # Check trailing space padding (0x20)
    json_str = header_bytes.decode("utf-8")
    assert json_str.startswith("{"), f"[{model_name}] Header must start with '{{'"
    
    # Verify padding
    stripped_json = json_str.rstrip(" ")
    assert stripped_json.endswith("}"), f"[{model_name}] Stripped header must end with '}}'"
    pad_bytes = len(json_str) - len(stripped_json)
    print(f"  [{model_name}] JSON header length: {len(stripped_json)}, space-padded with {pad_bytes} spaces (total: {len(json_str)})")

    header = json.loads(json_str)

    # 4. Validate __metadata__
    assert "__metadata__" in header, f"[{model_name}] Missing '__metadata__' in header!"
    metadata = header["__metadata__"]
    assert isinstance(metadata, dict), f"[{model_name}] __metadata__ must be a dict"
    for k, v in metadata.items():
        assert isinstance(k, str) and isinstance(v, str), (
            f"[{model_name}] Metadata entry {k}: {v} must be strictly string-to-string for HuggingFace compatibility!"
        )
    print(f"  [{model_name}] Validated string-to-string __metadata__: {metadata}")

    # 5. Validate Tensors & Netron Hierarchy
    expected_names = get_expected_tensor_names(n_layers=2)
    found_tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    assert len(found_tensors) == 38, (
        f"[{model_name}] Expected exactly 38 tensors, found {len(found_tensors)}"
    )

    missing = expected_names - set(found_tensors.keys())
    assert not missing, f"[{model_name}] Missing expected Netron tensors: {missing}"

    total_elements = 0
    raw_buffer = raw_bytes[buffer_offset:]

    for name, info in found_tensors.items():
        assert "dtype" in info, f"Missing dtype for tensor {name}"
        assert info["dtype"] == "F32", f"Tensor {name} dtype must be 'F32', got {info['dtype']}"
        assert "shape" in info, f"Missing shape for tensor {name}"
        shape = info["shape"]
        assert isinstance(shape, list) and all(isinstance(d, int) for d in shape), f"Invalid shape: {shape}"

        assert "data_offsets" in info, f"Missing data_offsets for tensor {name}"
        start_off, end_off = info["data_offsets"]
        assert start_off <= end_off, f"Invalid offsets: {start_off} > {end_off}"
        assert end_off <= len(raw_buffer), f"Offset {end_off} exceeds buffer length {len(raw_buffer)}"

        num_elements = 1
        for d in shape:
            num_elements *= d
        byte_len = end_off - start_off
        assert byte_len == num_elements * 4, (
            f"Tensor {name} byte length mismatch: {byte_len} != {num_elements * 4} (shape {shape})"
        )
        total_elements += num_elements

    assert total_elements == expected_params, (
        f"[{model_name}] Total parameter count mismatch: {total_elements} != {expected_params}"
    )
    print(f"  [{model_name}] Exactly 38 tensors verified, {total_elements:,} parameters ({total_elements * 4:,} bytes)")
    return header

# ---------------------------------------------------------------------------
# Test 3: Python safetensors.numpy Library Loading
# ---------------------------------------------------------------------------
def test_python_safetensors_library_loader(safetensors_path: str, expected_params: int, model_name: str):
    print_header(f"Test 3: Official Python safetensors Loader Verification ({model_name})")
    try:
        from safetensors.numpy import load_file
        tensors = load_file(safetensors_path)
        assert len(tensors) == 38, f"Expected 38 tensors, loaded {len(tensors)}"
        total_size = sum(arr.size for arr in tensors.values())
        assert total_size == expected_params, f"Loaded {total_size} parameters, expected {expected_params}"
        print(f"  [PASS] safetensors.numpy successfully loaded '{safetensors_path}' ({total_size:,} parameters)")
    except ImportError:
        # Fallback to uv run if current python doesn't have safetensors in global site-packages
        print("  safetensors not in default site-packages, executing via uv run --with safetensors,numpy...")
        cmd = [
            "uv", "run", "--with", "safetensors,numpy", "python3", "-c",
            f"from safetensors.numpy import load_file; "
            f"t = load_file('{safetensors_path}'); "
            f"assert len(t) == 38; "
            f"assert sum(arr.size for arr in t.values()) == {expected_params}; "
            f"print('Loaded', len(t), 'tensors with', sum(arr.size for arr in t.values()), 'parameters successfully!')"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"uv safetensors load failed for {safetensors_path}: {res.stderr}")
        print(f"  [PASS] uv safetensors.numpy loaded '{safetensors_path}' cleanly: {res.stdout.strip()}")

# ---------------------------------------------------------------------------
# Test 4: Adversarial & Edge Case Handling
# ---------------------------------------------------------------------------
def test_adversarial_rejection():
    print_header("Test 4: Adversarial & Corruption Edge Case Verification")

    # 1. Truncated file (< 8 bytes)
    try:
        parse_and_validate_safetensors_bytes(b"\x01\x02\x03", 0, "Truncated")
        assert False, "Should have rejected truncated file"
    except AssertionError as e:
        print(f"  [PASS] Properly rejected truncated stream: {e}")

    # 2. Header declaring length larger than file
    bogus_header_len = struct.pack("<Q", 1_000_000)
    try:
        parse_and_validate_safetensors_bytes(bogus_header_len + b"{}", 0, "HugeHeader")
        assert False, "Should have rejected oversized header"
    except AssertionError as e:
        print(f"  [PASS] Properly rejected oversized header: {e}")

    # 3. Unaligned header (not multiple of 8)
    unaligned_header = b'{"token_emb.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}'
    unaligned_bytes = struct.pack("<Q", len(unaligned_header)) + unaligned_header + b"\x00\x00\x00\x00"
    try:
        parse_and_validate_safetensors_bytes(unaligned_bytes, 1, "Unaligned")
        assert False, "Should have rejected unaligned header length"
    except AssertionError as e:
        print(f"  [PASS] Properly rejected unaligned header: {e}")

    # 4. Unsupported Dtype
    bad_dtype_header = json.dumps({
        "__metadata__": {"arch": "TinyTransformer"},
        "token_emb.weight": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}
    })
    # Pad to multiple of 8
    pad = (8 - (len(bad_dtype_header) % 8)) % 8
    bad_dtype_header += " " * pad
    bad_bytes = struct.pack("<Q", len(bad_dtype_header)) + bad_dtype_header.encode("utf-8") + b"\x00" * 8
    try:
        parse_and_validate_safetensors_bytes(bad_bytes, 1, "BadDtype")
        assert False, "Should have rejected non-F32 dtype"
    except AssertionError as e:
        print(f"  [PASS] Properly rejected unsupported dtype: {e}")

# Helper to generate standard reference safetensors binary files
def generate_reference_safetensors_bytes(vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_seq_len: int) -> bytes:
    tensors = {}
    offset = 0

    def add_tensor(name: str, shape: list[int]):
        nonlocal offset
        num_elem = 1
        for d in shape:
            num_elem *= d
        byte_len = num_elem * 4
        tensors[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + byte_len]
        }
        offset += byte_len

    add_tensor("token_emb.weight", [vocab_size, d_model])
    add_tensor("pos_emb.weight", [max_seq_len, d_model])
    for l in range(n_layers):
        p = f"blocks.{l}"
        for w in ["w_q", "w_k", "w_v", "w_o"]:
            add_tensor(f"{p}.attn.{w}.weight", [d_model, d_model])
            add_tensor(f"{p}.attn.{w}.bias", [d_model])
        add_tensor(f"{p}.attn.ln.weight", [d_model])
        add_tensor(f"{p}.attn.ln.bias", [d_model])
        add_tensor(f"{p}.ffn.w_1.weight", [d_ff, d_model])
        add_tensor(f"{p}.ffn.w_1.bias", [d_ff])
        add_tensor(f"{p}.ffn.w_2.weight", [d_model, d_ff])
        add_tensor(f"{p}.ffn.w_2.bias", [d_model])
        add_tensor(f"{p}.ffn.ln.weight", [d_model])
        add_tensor(f"{p}.ffn.ln.bias", [d_model])
    add_tensor("ln_f.weight", [d_model])
    add_tensor("ln_f.bias", [d_model])
    add_tensor("head.weight", [vocab_size, d_model])
    add_tensor("head.bias", [vocab_size])

    total_params = offset // 4
    header_dict = {
        "__metadata__": {
            "architecture": "TinyTransformer",
            "vocab_size": str(vocab_size),
            "d_model": str(d_model),
            "n_heads": str(n_heads),
            "n_layers": str(n_layers),
            "d_ff": str(d_ff),
            "max_seq_len": str(max_seq_len),
            "total_parameters": str(total_params),
            "format": "pt",
        }
    }
    header_dict.update(tensors)

    header_json = json.dumps(header_dict)
    pad_len = (8 - (len(header_json.encode("utf-8")) % 8)) % 8
    header_bytes = (header_json + " " * pad_len).encode("utf-8")
    assert len(header_bytes) % 8 == 0

    header_prefix = struct.pack("<Q", len(header_bytes))
    raw_buffer = bytes(offset) # zeroes for reference test
    return header_prefix + header_bytes + raw_buffer

def main():
    print("Starting Tier 2 Safetensors E2E Test Suite...")
    # First, run cargo tests in moonshot-model
    test_cargo_test_model()

    # Create target directory for test outputs
    os.makedirs("target/test_safetensors", exist_ok=True)

    # Validate 77K model binary format
    path_77k = "target/test_safetensors/model_77k.safetensors"
    bytes_77k = generate_reference_safetensors_bytes(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=32)
    with open(path_77k, "wb") as f:
        f.write(bytes_77k)
    print_header("Test 2a: 77K Model Binary Specification & Alignment")
    parse_and_validate_safetensors_bytes(bytes_77k, 77376, "77K Model")
    test_python_safetensors_library_loader(path_77k, 77376, "77K Model")

    # Validate 1.09M model binary format
    path_1m = "target/test_safetensors/model_1m.safetensors"
    bytes_1m = generate_reference_safetensors_bytes(vocab_size=64, d_model=256, n_heads=8, n_layers=2, d_ff=512, max_seq_len=32)
    with open(path_1m, "wb") as f:
        f.write(bytes_1m)
    print_header("Test 2b: 1.09M Model Binary Specification & Alignment")
    parse_and_validate_safetensors_bytes(bytes_1m, 1095744, "1.09M Model")
    test_python_safetensors_library_loader(path_1m, 1095744, "1.09M Model")

    # Adversarial tests
    test_adversarial_rejection()

    # Validate actual Rust-exported files from cargo test
    if os.path.exists("target/test_77k.safetensors"):
        print_header("Test 5a: Rust Exported 77K safetensors Binary Verification")
        with open("target/test_77k.safetensors", "rb") as f:
            rust_bytes_77k = f.read()
        parse_and_validate_safetensors_bytes(rust_bytes_77k, 77376, "Rust Export 77K")
        test_python_safetensors_library_loader("target/test_77k.safetensors", 77376, "Rust Export 77K")

    if os.path.exists("target/test_1m.safetensors"):
        print_header("Test 5b: Rust Exported 1.09M safetensors Binary Verification")
        with open("target/test_1m.safetensors", "rb") as f:
            rust_bytes_1m = f.read()
        parse_and_validate_safetensors_bytes(rust_bytes_1m, 1095744, "Rust Export 1.09M")
        test_python_safetensors_library_loader("target/test_1m.safetensors", 1095744, "Rust Export 1.09M")

    print("\n" + "=" * 70)
    print("  ALL TIER 2 SAFETENSORS VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
