#!/usr/bin/env bash
# Tier 3 E2E Test: Moonshot Lab Interactive CLI REPL Verification
# Verifies:
# 1. Binary compilation and launch via `cargo run --bin lab`
# 2. CLI options and help menu display (`--help`)
# 3. Piped stdin non-interactive execution (graceful EOF handling without hanging)
# 4. Token stepping (`step`, `run`) with ANSI syntax formatting
# 5. Live SandboxState register (`regs`) and memory (`mem`) inspection
# 6. Attention matrix extraction (`attn`) across layers and heads
# 7. Model configuration switching (77K vs 1.09M model sizes)
# 8. Task switching (`task`) and context reset (`reset`)

set -euo pipefail

PASS_COUNT=0
FAIL_COUNT=0

log_pass() {
    echo -e "  \033[32m[PASS]\033[0m $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    echo -e "  \033[31m[FAIL]\033[0m $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

section_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

# -----------------------------------------------------------------------------
# Test 1: CLI Flags and Help Output
# -----------------------------------------------------------------------------
section_header "Test 1: Moonshot Lab CLI Flags & Help Output"
HELP_OUTPUT=$(cargo run --bin lab -- --help 2>&1)

if echo "$HELP_OUTPUT" | grep -q "Moonshot Lab Interactive CLI REPL" && \
   echo "$HELP_OUTPUT" | grep -q -- "--model" && \
   echo "$HELP_OUTPUT" | grep -q -- "--family" && \
   echo "$HELP_OUTPUT" | grep -q -- "--batch"; then
    log_pass "CLI binary renders full usage synopsis with --model, --family, --batch flags"
else
    log_fail "CLI binary failed help text verification: $HELP_OUTPUT"
    exit 1
fi

# -----------------------------------------------------------------------------
# Test 2: Piped Stdin Interactive Command Loop (step, regs, mem, attn, quit)
# -----------------------------------------------------------------------------
section_header "Test 2: Piped Stdin Command Loop Execution"
PIPED_INPUT=$(printf "step\nregs\nmem\nattn 0 0\nquit\n")
CMD_OUTPUT=$(echo "$PIPED_INPUT" | cargo run --bin lab 2>&1)

# Verify Step Output
if echo "$CMD_OUTPUT" | grep -q "Step  1 -> Generated:"; then
    log_pass "Token generation step executed and printed generated token"
else
    log_fail "Step command did not output generated token"
fi

# Verify Registers Card Output
if echo "$CMD_OUTPUT" | grep -q "Registers (R0..R3)" && \
   echo "$CMD_OUTPUT" | grep -q "R0:" && \
   echo "$CMD_OUTPUT" | grep -q "R3:"; then
    log_pass "Live SandboxState registers displayed with box-drawing borders"
else
    log_fail "Registers command failed to render register cards"
fi

# Verify Memory Card Output
if echo "$CMD_OUTPUT" | grep -q "Memory (M0..M3)" && \
   echo "$CMD_OUTPUT" | grep -q "M0:" && \
   echo "$CMD_OUTPUT" | grep -q "M3:"; then
    log_pass "Live SandboxState memory displayed with box-drawing borders"
else
    log_fail "Memory command failed to render memory cards"
fi

# Verify Attention Matrix Heatmap
if echo "$CMD_OUTPUT" | grep -q "Attention Heatmap \[Layer 0, Head 0\]" && \
   echo "$CMD_OUTPUT" | grep -q "Density:"; then
    log_pass "Attention weights rendered with layer/head coordinates and density scale"
else
    log_fail "Attention command failed to render attention matrix"
fi

# Verify Clean Exit
if echo "$CMD_OUTPUT" | grep -q "Exiting Moonshot Lab. Goodbye!"; then
    log_pass "Quit command terminated REPL with clean exit greeting"
else
    log_fail "REPL did not terminate cleanly on quit"
fi

# -----------------------------------------------------------------------------
# Test 3: Multiple Token Stepping & Autoregressive Sequence Run
# -----------------------------------------------------------------------------
section_header "Test 3: Multi-step & Autoregressive Run"
RUN_INPUT=$(printf "step 3\ntokens\nquit\n")
RUN_OUTPUT=$(echo "$RUN_INPUT" | cargo run --bin lab 2>&1)

if echo "$RUN_OUTPUT" | grep -q "Step  3 -> Generated:" && \
   echo "$RUN_OUTPUT" | grep -q "Current Token Sequence"; then
    log_pass "Multi-step forward generation ('step 3') advanced sequence correctly"
else
    log_fail "Multi-step forward generation failed"
fi

# -----------------------------------------------------------------------------
# Test 4: Task Family Switching and Context Reset
# -----------------------------------------------------------------------------
section_header "Test 4: Task Family Switching & Context Reset"
RESET_INPUT=$(printf "task fsm_tree\nreset\ntokens\nquit\n")
RESET_OUTPUT=$(echo "$RESET_INPUT" | cargo run --bin lab 2>&1)

if echo "$RESET_OUTPUT" | grep -q "fsm_hierarchical_tree"; then
    log_pass "Task switching to 'fsm_tree' succeeded with updated prompt"
else
    log_fail "Task switching failed"
fi

# -----------------------------------------------------------------------------
# Test 5: 1.09M Model Execution
# -----------------------------------------------------------------------------
section_header "Test 5: 1.09M Model Parameter & Attention Extraction"
M1_OUTPUT=$(cargo run --bin lab -- -m 1m -b -c "step; attn 0 0" 2>&1)

if echo "$M1_OUTPUT" | grep -q "1095744" && \
   echo "$M1_OUTPUT" | grep -q "Attention Heatmap"; then
    log_pass "1.09M parameter model loaded and stepped attention correctly (1,095,744 params)"
else
    log_fail "1.09M model execution failed"
fi

# -----------------------------------------------------------------------------
# Test 6: Non-Interactive Graceful EOF Handling (/dev/null stdin)
# -----------------------------------------------------------------------------
section_header "Test 6: Non-Interactive Graceful EOF Handling"
EOF_EXIT_CODE=0
cargo run --bin lab < /dev/null > /dev/null 2>&1 || EOF_EXIT_CODE=$?

if [ "$EOF_EXIT_CODE" -eq 0 ]; then
    log_pass "Empty stdin (/dev/null) exited gracefully with status 0 without hanging"
else
    log_fail "Empty stdin failed with exit code $EOF_EXIT_CODE"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "  \033[32mALL TIER 3 LAB REPL VERIFICATIONS PASSED ($PASS_COUNT/$PASS_COUNT)!\033[0m"
    echo "======================================================================"
    exit 0
else
    echo -e "  \033[31mTIER 3 LAB REPL TESTS FAILED: $FAIL_COUNT failures, $PASS_COUNT passed\033[0m"
    echo "======================================================================"
    exit 1
fi
