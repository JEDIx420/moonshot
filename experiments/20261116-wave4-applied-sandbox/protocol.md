# Pre-Registration Protocol: Wave 4 (Applied Multi-Step Tool Agent Sandbox / Leap D)

**Experiment ID**: `20261116-wave4-applied-sandbox`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`

---

## 1. Primary Hypothesis (Hypothesis D)

In a simulated, safe multi-step tool agent sandbox requiring variable assignment, register arithmetic, conditional branching, and recovery from simulated hardware faults:

1. **Hypothesis D1 (Legal Execution Traces)**: The learned credit routing / plasticity-gated learner generates valid, executable tool sequences that strictly adhere to sandbox grammar and operational semantics at a rate $\ge 0.70$, decisively outperforming the unadapted base model ($\le 0.25$) and static heuristic layer freezing.
2. **Hypothesis D2 (Final World State Concordance)**: When executed in the sandbox simulator, generated action sequences bring the simulated environment into the exact target world state with $\ge 0.50$ accuracy on unexposed task instances.
3. **Hypothesis D3 (Sample Efficiency Advantage)**: Plasticity routing achieves criterion competence ($\ge 80\%$ token accuracy / state reachability) using $\ge 25\%$ fewer gradient steps and sample exposures compared to standard AdamW without gating.
4. **Hypothesis D4 (Sandbox Containment)**: 100% of execution occurs in pure, isolated Rust in-memory state machines with zero external network, disk, or host system side effects.

---

## 2. Sandbox Environment & Grammar Specification

The sandbox models a simulated register machine with:
- 4 General Registers: $R_0, R_1, R_2, R_3 \in [0, 15]$
- 4 Memory Slots: $M_0, M_1, M_2, M_3 \in [0, 15]$
- Error / Exception Flag: `FLAG_ERR`
- Program Counter / Step Limit: 16 operations max per episode

### Action Tokens & Grammar:
1. `CMD_TOOL_SET reg val` ($3$ tokens): Sets $R_{\text{reg}} \leftarrow \text{val}$.
2. `CMD_TOOL_ADD src1 src2 dst` ($4$ tokens): Sets $R_{\text{dst}} \leftarrow (R_{\text{src1}} + R_{\text{src2}}) \pmod{16}$.
3. `CMD_TOOL_COPY src dst` ($3$ tokens): Sets $M_{\text{dst}} \leftarrow R_{\text{src}}$.
4. `CMD_TOOL_BRANCH reg offset` ($3$ tokens): If $R_{\text{reg}} == 0$, skip `offset` instructions.
5. `CMD_TOOL_RECOVER err_code` ($2$ tokens): Clears `FLAG_ERR` and resets destination register.
6. `CMD_TOOL_COMMIT` ($1$ token): Verifies world state invariants and terminates execution.

A sequence is **Legal** if and only if:
- Every opcode is followed by the exact expected number of valid operands within range $[0, 15]$.
- No undefined registers are read prior to initialization.
- Execution terminates with a valid `CMD_TOOL_COMMIT` followed by `EOS_TOKEN`.

---

## 3. Experimental Factorial Design & Baselines

Evaluated on $N=3$ unopened test seeds (`101, 202, 303`) across 5 comparative adaptation conditions:
1. `FrozenBaseline`: Zero-shot base model execution (0 adaptation steps).
2. `FixedAdamW`: Standard AdamW optimization ($\alpha = 0.001, 55$ steps) with all layers fully plastic.
3. `LearnedPlasticity`: Wave 2 learned credit router ($\bar{\gamma} = 0.661$, freezing early representation drift while optimizing task heads).
4. `StaticHeadOnly`: Fixed architectural heuristic freezing all layers except prediction head.
5. `RandomPlasticity`: Stochastic layer gating matched in average plastic parameter fraction.

---

## 4. Verification Gates

- **Gate D1 (Trace Legality Rate)**: $\text{LegalRate}(\text{LearnedPlasticity}) \ge 0.70$.
- **Gate D2 (Final State Accuracy)**: $\text{StateAcc}(\text{LearnedPlasticity}) \ge 0.50$ across held-out sequences.
- **Gate D3 (Sample Efficiency Gain)**: $\text{StepsToTarget}(\text{LearnedPlasticity}) \le 0.75 \times \text{StepsToTarget}(\text{FixedAdamW})$.
- **Gate D4 (Containment & Determinism)**: Zero host side effects, bitwise idempotent state reproduction across repeated simulation runs.
