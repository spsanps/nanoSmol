# SmolLM2 Test Utilities

The scripts in this folder are intentionally lightweight inspection tools for
verifying that the minimal `models/smolLM2/` implementation stays aligned with
Hugging Face's reference checkpoints.  They are not meant to be exhaustive unit
tests, but rather interactive debugging helpers.

## Known debug script behaviour

`debug_layer0_breakdown.py` performs a stage-by-stage comparison between the
first decoder block of the Hugging Face model and the local implementation. The
script prints a `False` line at the top because it currently dumps the value of
`hf.config.attention_bias` (which is unset for SmolLM2).  The expected output
when running against `HuggingFaceTB/SmolLM2-135M` with `--dtype float32` looks
like this:

```
False
has_q_bias False
has_k_bias False
has_v_bias False
[cfg] d_model=576 layers=30 heads=9 kv_heads=3 ffn=1536 rope_theta=100000.0 pad_id=0
[load_hf] src layers detected: 0..29 (count=30) ; dst expects 30
[load_hf] copied tensors: 272
[load_hf] unused source tensors: 0 (showing up to 10) -> []
Δ q_proj.w[0] 0.0
Δ k_proj.w[0] 0.0
Δ v_proj.w[0] 0.0
LN1 max Δ: 0.0
Q lin max Δ: 0.0
K lin max Δ: 0.0
stage diffs (max abs):
LN1          max|Δ|=0
Q (all)      max|Δ|=4.16137
K (all)      max|Δ|=9.45432
V (all)      max|Δ|=0
Q_rope       max|Δ|=0
K_rope       max|Δ|=0
AttnOut      max|Δ|=0
O_proj       max|Δ|=0
Resid1       max|Δ|=0
LN2          max|Δ|=0
MLP(out)     max|Δ|=0
Layer0 out   max|Δ|=0
```

(The large `Q`/`K` diffs simply reflect how the HF module stores the projections
before reshaping; downstream values still match.)

The other scripts (`compare_hf_vs_local.py`, `dump_first_diff_layer.py`, and
`generate_smoke.py`) continue to work as small parity and sampling tools.  They
now import the refactored package layout without modification.
