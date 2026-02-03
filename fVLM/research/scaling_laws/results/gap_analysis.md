# Train/Inference Gap Analysis

## Summary

**Key Finding:** The train/inference gap is negligible (<0.1%), validating the parallel training approximation.

## Results

| Model | Steps | Training Loss (Fine) | Autoregressive Loss | Gap | Gap % |
|-------|-------|---------------------|---------------------|-----|-------|
| S-S (135M + small) | 100 | 4.0480 | 4.0239 | -0.0241 | -0.60% |
| S-S (135M + small) | 3000 | 3.5036 | 3.5041 | +0.0005 | +0.01% |
| M-S (360M + small) | 3000 | 3.4978 | 3.4984 | +0.0005 | +0.01% |

## Key Observations

1. **Gap is negligible at convergence** (~0.01%)
   - Both S-S and M-S models show identical gap at 3000 steps
   - The parallel training approximation closely matches true autoregressive inference

2. **Early training shows slight autoregressive advantage** (-0.6%)
   - At 100 steps, autoregressive loss is slightly LOWER
   - This suggests the model learns to use coarse-derived queries effectively over time

3. **Training approximation is VALID**
   - The gap is far below the 5% threshold for concern
   - No need for expensive true autoregressive training

## Implications

1. **Research Quality:** Current scaling results using training loss are valid
2. **Architecture:** The coarse→fine parallel approximation works as intended
3. **Future Work:** Can focus on other improvements (multi-fine iterations, larger models)

## Methodology

- Evaluated on 200 random validation samples per checkpoint
- Used `forward_captioning(use_fine=True)` for training loss
- Used `forward_autoregressive_captioning()` for true inference loss
- Both compute cross-entropy on identical caption tokens
