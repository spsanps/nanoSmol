# Scaling Law Results Summary

## Raw Data

| Config | Type | Step | Loss | ±SE | PPL | FLOPs (P) | Visual Tokens |
|--------|------|------|------|-----|-----|-----------|---------------|
| M-S | baseline | 100 | 4.1244 | 0.0648 | 61.83 | 3.30 | 16 |
| M-S | baseline | 300 | 3.9363 | 0.0633 | 51.23 | 9.90 | 16 |
| M-S | baseline | 1000 | 3.7767 | 0.0658 | 43.67 | 32.99 | 16 |
| M-S | baseline | 3000 | 3.6431 | 0.0635 | 38.21 | 98.96 | 16 |
| M-S | foveated | 100 | 4.1842 | 0.0618 | 65.64 | 2.05 | 1 |
| M-S | foveated | 300 | 4.0410 | 0.0607 | 56.88 | 6.16 | 1 |
| M-S | foveated | 1000 | 3.9123 | 0.0623 | 50.02 | 20.54 | 1 |
| M-S | foveated | 3000 | 3.7891 | 0.0629 | 44.22 | 61.63 | 1 |
| S-S | baseline | 100 | 4.3678 | 0.0630 | 78.87 | 2.05 | 16 |
| S-S | baseline | 300 | 4.0656 | 0.0629 | 58.30 | 6.14 | 16 |
| S-S | baseline | 1000 | 3.8719 | 0.0631 | 48.03 | 20.48 | 16 |
| S-S | baseline | 3000 | 3.7571 | 0.0665 | 42.82 | 61.44 | 16 |
| S-S | foveated | 100 | 4.4617 | 0.0611 | 86.64 | 1.58 | 1 |
| S-S | foveated | 300 | 4.1770 | 0.0586 | 65.17 | 4.74 | 1 |
| S-S | foveated | 1000 | 4.0188 | 0.0617 | 55.63 | 15.82 | 1 |
| S-S | foveated | 3000 | 3.8608 | 0.0626 | 47.51 | 47.45 | 1 |

## Key Findings

### At 3000 Steps (Final)

- **S-S**: Foveated loss 3.8608 vs Baseline 3.7571 (+2.76% higher)
  - Foveated uses 1 tokens, Baseline uses 16 tokens
  - Efficiency ratio: 16x fewer tokens for 2.8% worse loss
- **M-S**: Foveated loss 3.7891 vs Baseline 3.6431 (+4.01% higher)
  - Foveated uses 1 tokens, Baseline uses 16 tokens
  - Efficiency ratio: 16x fewer tokens for 4.0% worse loss

### Model Size Effect

- **Foveated**: M-S (360M) vs S-S (135M) = -1.86% (improvement)
- **Baseline**: M-S (360M) vs S-S (135M) = -3.03% (improvement)
