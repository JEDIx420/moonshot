#!/usr/bin/env bash
# ==============================================================================
# Moonshot Master E2E Test Suite Runner
# Executes independent verification across all four tiers:
#   Tier 1: Micro-Autodiff Engine Numerical Verification (tests/test_autodiff_e2e.py)
#   Tier 2: Safetensors Model Serialization & Python Loading (tests/test_safetensors_python.py)
#   Tier 3: Moonshot Lab REPL Interactive Stepping & Inspection (tests/test_lab_repl_e2e.sh)
#   Tier 4: Moonshot Cockpit Dashboard REST API & UI (tests/test_cockpit_e2e.py)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║            MOONSHOT MASTER E2E TEST SUITE RUNNER                       ║${RESET}"
echo -e "${BOLD}${CYAN}║  4-Tier Multi-Tool Independent Opaque-Box Verification Harness          ║${RESET}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════╝${RESET}"

TOTAL_TIERS=4
PASSED_TIERS=0
FAILED_TIERS=0

TIER1_STATUS="PENDING"
TIER2_STATUS="PENDING"
TIER3_STATUS="PENDING"
TIER4_STATUS="PENDING"

START_TIME=$(date +%s)

run_tier() {
    local tier_num="$1"
    local tier_name="$2"
    local cmd="$3"

    echo ""
    echo -e "${BOLD}▶ [Tier ${tier_num}/4] Starting ${tier_name}...${RESET}"
    local tier_start=$(date +%s)
    local exit_code=0
    eval "$cmd" || exit_code=$?
    local tier_end=$(date +%s)
    local tier_duration=$((tier_end - tier_start))

    if [ "$exit_code" -eq 0 ]; then
        echo -e "${BOLD}${GREEN}✔ [Tier ${tier_num}/4] ${tier_name} PASSED in ${tier_duration}s${RESET}"
        PASSED_TIERS=$((PASSED_TIERS + 1))
        return 0
    else
        echo -e "${BOLD}${RED}✘ [Tier ${tier_num}/4] ${tier_name} FAILED with exit code ${exit_code} after ${tier_duration}s${RESET}"
        FAILED_TIERS=$((FAILED_TIERS + 1))
        return 1
    fi
}

# Tier 1: Micro-Autodiff Engine
if run_tier 1 "Micro-Autodiff Engine Numerical Verification" "python3 tests/test_autodiff_e2e.py"; then
    TIER1_STATUS="PASSED"
else
    TIER1_STATUS="FAILED"
fi

# Tier 2: Safetensors Model Exporter/Importer
if run_tier 2 "Safetensors Exporter/Importer & Python Loader" "python3 tests/test_safetensors_python.py"; then
    TIER2_STATUS="PASSED"
else
    TIER2_STATUS="FAILED"
fi

# Tier 3: Moonshot Lab REPL
if run_tier 3 "Moonshot Lab REPL Interactive Stepping" "bash tests/test_lab_repl_e2e.sh"; then
    TIER3_STATUS="PASSED"
else
    TIER3_STATUS="FAILED"
fi

# Tier 4: Moonshot Cockpit Dashboard
if run_tier 4 "Moonshot Cockpit Dashboard REST API & UI" "python3 tests/test_cockpit_e2e.py"; then
    TIER4_STATUS="PASSED"
else
    TIER4_STATUS="FAILED"
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "${BOLD}${CYAN}                      E2E TEST SUMMARY REPORT                         ${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "  Tier 1 [Autodiff Engine]:     $([ "$TIER1_STATUS" == "PASSED" ] && echo -e "${GREEN}PASSED${RESET}" || echo -e "${RED}FAILED${RESET}")"
echo -e "  Tier 2 [Safetensors]:         $([ "$TIER2_STATUS" == "PASSED" ] && echo -e "${GREEN}PASSED${RESET}" || echo -e "${RED}FAILED${RESET}")"
echo -e "  Tier 3 [Lab REPL]:            $([ "$TIER3_STATUS" == "PASSED" ] && echo -e "${GREEN}PASSED${RESET}" || echo -e "${RED}FAILED${RESET}")"
echo -e "  Tier 4 [Cockpit Dashboard]:   $([ "$TIER4_STATUS" == "PASSED" ] && echo -e "${GREEN}PASSED${RESET}" || echo -e "${RED}FAILED${RESET}")"
echo -e "  Total Duration:               ${TOTAL_DURATION}s"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"

if [ "$FAILED_TIERS" -eq 0 ]; then
    echo -e "${BOLD}${GREEN}✔ ALL 4 TIERS PASSED INDEPENDENT VERIFICATION (4/4)!${RESET}"
    exit 0
else
    echo -e "${BOLD}${RED}✘ VERIFICATION FAILED: ${FAILED_TIERS} tiers failed, ${PASSED_TIERS} passed.${RESET}"
    exit 1
fi
