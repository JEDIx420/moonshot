# Wave 6A Findings: 1.09M Model Verification & Baseline

**Audit Date**: 2026-11-17T03:08:05.633643497+00:00  
- **Total Parameters**: 1,095,744 ($d_{\text{model}}=256, L=2, H=8, d_{\text{ff}}=512$)
- **Finite-Difference Gradient Error**: 1.2801 (Tolerance: < 0.05)
- **Checkpoint Restoration**: Bitwise exact (PASSED)
- **1M Mean Exact Match**: 7.78% (vs 77K: 45.00%)
- **Gate S0**: FAILED
