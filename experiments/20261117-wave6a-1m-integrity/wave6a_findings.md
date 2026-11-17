# Wave 6A Findings: 1.09M Model Verification & Baseline

**Audit Date**: 2026-11-17T03:52:37.357640813+00:00  
- **Total Parameters**: 1,095,744 ($d_{\text{model}}=256, L=2, H=8, d_{\text{ff}}=512$)
- **Finite-Difference Gradient Error**: 1.2801 (Significant Error: 0.0000, Discrepancies: 0)
- **Checkpoint Restoration**: Bitwise exact (PASSED)
- **1M Single-Task Exact Match (`ToolWorkflowBasic`)**: 48.33% (vs 77K: 96.67%)
- **1M Multi-Step Sandbox Baseline (`ToolWorkflowSandbox`)**: 0.00% (vs 77K: 61.67%)
- **Gate S0**: FAILED
