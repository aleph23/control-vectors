# Model support notes (deferred work)

Context: `model_handler.py`'s `ModelHandler.__init__` currently hardcodes a single
architecture check (`isGemma3`) to decide `attn_implementation` and dtype overrides.
This doesn't scale as Transformers adds new architectures (Gemma4, Qwen3.6, etc.).
Deferred pending scope — see chat history around 2026-07-30 for the full discussion.

## Planned direction (not yet implemented)

1. Replace manual `json.load(config.json)` parsing with `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
   - Generalizes across architectures without hand-tracking key names/shapes.
2. Stop forcing `attn_implementation="flash_attention_2"` by default; pass `None` and let
   Transformers auto-select (flash_attention_2 → sdpa → eager) with a safe fallback. This needs to be double-checked.  Transformers default is presently SDPA-first.  flash_attn might need internal implementation.
3. Replace the `isGemma3` branch with a small quirks registry keyed by architecture name
   (`config.architectures[0]`), storing only verified overrides:
   ```python
   _ARCH_QUIRKS = {
       "Gemma3ForCausalLM": {"attn_implementation": "eager", "avoid_float16": True},
       # "Gemma4ForCausalLM": {...},  # add once actually verified needed (see notes below)
   }
   ```
   Adding a new model's support is then a one-line dict entry (or nothing, if it needs
   no special handling) rather than a new code branch.
4. Offload `_ARCH_QUIRKS` from a hardcoded Python dict into a YAML file (e.g.
   `model_quirks.yaml`) shipped alongside `model_handler.py`. This lets end users add or
   override entries for new/unreleased architectures without editing source — just drop
   a new mapping in the YAML and it's picked up on next run. Keeps the same shape as the
   dict above (architecture name → override keys); `model_handler.py` would just
   `yaml.safe_load()` it once at import/init time instead of defining the dict inline.
   Worth deciding whether unknown/user-supplied keys should be validated against a known
   set (`attn_implementation`, `avoid_float16`, ...) or passed through permissively.

## Findings from `modular_gemma4.py` (transformers, as of 2026-07)

Source: HF `transformers` package, `models/gemma4/modular_gemma4.py` (Apache-2.0).

- **Class hierarchy**: `Gemma4ForCausalLM` subclasses `Gemma3ForCausalLM`;
  `Gemma4Model`/`Gemma4TextModel` subclass `Gemma3nModel`/`Gemma3TextModel` (note:
  Gemma3**n**, the multimodal variant, not plain Gemma3). `Gemma4PreTrainedModel`
  subclasses `Gemma3nPreTrainedModel`. So Gemma4 is architecturally closer to the
  Gemma3n multimodal line than to text-only Gemma3.
- **Multimodal by default**: `Gemma4Config` composes `Gemma4TextConfig` +
  `Gemma4VisionConfig` + `Gemma4AudioConfig`. `input_modalities = ("image", "text",
  "video", "audio")`. This means for Gemma4, `config.json`'s top-level
  `architectures[0]` may be `"Gemma4ForConditionalGeneration"` (multimodal) rather than
  `"Gemma4ForCausalLM"` (text-only) — our quirks-registry lookup key needs to account
  for both, and `model_type` substring matching (like our current `"gemma3" in
  model_type.lower()`) is probably more robust than exact architecture-string matches
  for this family, mirroring what we already do for Gemma3.
- **Attention implementation**: Gemma4 attention modules (`Gemma4TextAttention`,
  `Gemma4VisionAttention`) resolve their backend generically via
  `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
  eager_attention_forward)` — i.e. Gemma4 itself doesn't hardcode eager-only behavior
  in the modeling code the way older Gemma releases needed external workarounds for.
  This suggests Gemma4 may not need the same forced `"eager"` override we use for
  Gemma3 — but this needs to be verified empirically (e.g. does flash_attention_2 or
  sdpa actually work correctly on a real Gemma4 checkpoint?) before assuming so in the
  quirks table.
- **New structural features not present in Gemma3** (context for why "just reuse the
  Gemma3 quirks" is risky):
  - Mixture-of-experts blocks (`Gemma4TextRouter`, `Gemma4TextExperts`,
    `enable_moe_block`) on later layers.
  - KV-sharing across layers (`is_kv_shared_layer`, `shared_kv_states`) — later layers
    can reuse an earlier layer's K/V instead of computing their own.
  - Per-Layer Embeddings (PLE): an auxiliary embedding path
    (`embed_tokens_per_layer`, `get_per_layer_inputs`) feeding a residual signal into
    each decoder layer, on top of the normal token embedding.
  - `Gemma4ClippableLinear`: optional activation clipping (`use_clipped_linears`) on
    vision/audio projections — a numerical-stability feature that could matter if we
    ever touch those weights directly (we don't, today — `modify_tensor` only touches
    `mlp.down_proj` on text decoder layers).
  - Multidimensional RoPE for vision (`apply_multidimensional_rope`) and a custom
    `Gemma4TextRotaryEmbedding` with per-layer-type (`full_attention` vs
    `sliding_attention`) inverse frequencies.
- **dtype/precision-sensitive spots seen in the modeling code** (useful if we ever see
  numerical issues to debug): vision pooler scaling (`sqrt(hidden_size)`) is
  deliberately done in float32 because it "can exceed the float16 range", audio
  attention logit softcap/clamping also runs in float32. This is consistent with
  Gemma-family models generally disliking float16 — supports keeping/extending our
  existing `avoid_float16`-style override for this family rather than assuming Gemma4
  is float16-safe just because the attention code looks more generic.
- **`get_num_layers()` / `modify_tensor()` compatibility**: our current
  `model_handler.py` assumes `self.model.model.layers` (a flat decoder stack). For a
  multimodal Gemma4 (`Gemma4ForConditionalGeneration`), the layer stack lives at
  `self.model.model.language_model.layers` (text decoder) with separate
  `vision_tower`/`audio_tower` submodules alongside it — `get_num_layers()`/
  `modify_tensor()` will need a structural update (not just a quirks-table entry) to
  support multimodal Gemma4 checkpoints, since `hasattr(self.model.model, 'layers')`
  will be `False` for those.

## Open questions to resolve before implementing

- Does a real Gemma4 checkpoint actually need `attn_implementation="eager"`, or does
  its generic attention-interface lookup mean flash_attention_2/sdpa work fine now?
  (Needs empirical testing against an actual released checkpoint, not just code
  reading.)
- Do we ever want/need to target Gemma4's multimodal variant for control-vector /
  conceptor extraction, or only the text-only `Gemma4ForCausalLM` path? This decides
  whether `get_num_layers()`/`modify_tensor()` need the structural update above.
- Same open questions apply to Qwen3.6 and any other new architecture — the quirks
  registry only helps once we've actually verified what (if anything) a given
  architecture needs; it shouldn't be pre-populated with guesses.

## GGUF as a model *source* (not just export target) — considered, deferred

Transformers can load `.gguf` checkpoints directly (`AutoModelForCausalLM.from_pretrained(
 path, gguf_file=...)`), but it works by **dequantizing** the GGUF tensors back into a
regular fp16/bf16 `nn.Module` in memory. Considered as a way to let `ModelHandler`
accept a raw `.gguf` file as `pretrained_model_name_or_path` instead of requiring a full
HF-format (safetensors) directory.

**Conclusion: not worth adding as a general capability** — dequantizing a GGUF checkpoint
just to re-probe it introduces quantization noise into the captured hidden states with no
compensating benefit (no memory savings vs. a safetensors checkpoint once dequantized,
and GGUF architecture/quant-type coverage in Transformers lags new releases anyway — see
above re: Gemma4/Qwen3.6). Adding it as a generic input path would mean every conceptor/
direction-vector extraction silently inherits whatever quantization error the source GGUF
already baked in, for models where we usually *do* have a non-quantized source available.

**Where it *does* make sense**: gated behind `--precision orig`. That flag already means
"trust the checkpoint's own dtype/quantization as-is, don't second-guess it" — a user who
picks `orig` is explicitly opting into whatever precision the source checkpoint already
has. So if `pretrained_model_name_or_path` points at a `.gguf` file and `precision ==
"orig"`, loading it via Transformers' `gguf_file=` dequantization path is consistent with
that choice rather than introducing a surprise: the user already said "don't touch the
precision," and GGUF's baked-in quantization error is just what "the checkpoint's own
dtype" means in that case. This should NOT be wired up for `"bfloat16"`/`"4bit"`/`"8bit"`
— those imply the user wants us to actively manage precision, and stacking our own
quantization/dtype choices on top of an already-dequantized-from-quantized model would
compound noise for no benefit.

Implementation sketch (not yet done): in `ModelHandler.__init__`, detect a `.gguf` path
(e.g. `str(pretrained_model_name_or_path).endswith(".gguf")`), and when combined with
`precision == "orig"`, branch to `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
gguf_file=pretrained_model_name_or_path, ...)` — note there's no separate `config.json` to
read in this case, read the gguf file's own internal config. Reject
(raise) the combination of a `.gguf` source path with `precision in ("bfloat16", "4bit",
"8bit")` rather than silently reinterpreting the user's intent or ignore precision flag altogether and use what we are given.

## Target platform: DGX Spark (GB10, aarch64/sbsa)

The real target for this pipeline is a DGX Spark (Grace-Blackwell GB10, aarch64/sbsa,
CUDA), not x86. Two CLI options depend on packages whose aarch64 availability is not a
given.

### PyTorch

PyTorch publishes `manylinux_aarch64` wheels through the standard CUDA index URLs.
`torch>=2.12.0` as pinned in `requirements.txt` is obtainable on aarch64 via:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Use the CUDA version matching the DGX Spark's driver — check `nvidia-smi` on the
target. The `cu128` index is the likely match for a Blackwell-era system.)

**Status:** verified from PyTorch's published wheel index (https://pytorch.org/get-started/locally/,
checked 2026-08-12). The `manylinux` wheel tag covers aarch64 with glibc ≥ 2.28.

### bitsandbytes (`--precision 4bit` / `--precision 8bit`)

bitsandbytes publishes a `manylinux_2_24_aarch64` wheel as of version 0.50.0 (checked
2026-08-12 on PyPI). The system requirements table on the project page confirms aarch64
Linux support with NVIDIA GPU (CUDA, SM75+). GB10's Blackwell GPU should be well above
this floor.

**Status:** verified from published PyPI wheels. `pip install bitsandbytes` should
work on the DGX Spark without a source build. Fallback if unavailable: use
`--precision bfloat16` (which routes through `torch.bfloat16` without any bnb
dependency — see `model_handler.py:36-41`).

### flash-attn (`--attn flash`)

flash-attn publishes **no pre-built wheels for any platform** as of version 2.8.3.post1
(checked 2026-08-12 on PyPI). Only a source distribution (`flash_attn-2.8.3.post1.tar.gz`)
is available. Building from source requires:

- A CUDA toolchain (nvcc)
- Sufficient RAM (the build is resource-intensive)
- The target GPU's compute capability must be supported

On aarch64/SBSA, the build situation is additionally complicated by the non-x86
host toolchain. **Recommended:** use `--attn sdpa` or `--attn none` (the default) on
the DGX Spark. SDPA is PyTorch's built-in fused attention and is available on all
CUDA platforms without extra dependencies.

If `--attn flash` is required, install `requirements-flash.txt` and expect a
from-source build. See `requirements-flash.txt` for the install command.

**Status:** inferred from PyPI wheel availability. No aarch64 wheel exists; source
build is the only path. This has not been empirically tested on a GB10.

### Unified memory

The DGX Spark's GB10 uses unified LPDDR5X memory shared between the Grace CPU and
Blackwell GPU. This changes the calculus behind the upstream pattern of "reload the
model on CPU to free VRAM" — on a unified-memory system, a CPU-side reload does not
free any physical memory; it just changes the access path. This branch therefore loads
on `cuda` throughout (see `create_control_vectors.py` — `torch.set_default_device("cuda")`
and all `ModelHandler(..., device="cuda")` calls). The `free_memory()` call between
the analyzer and the export step is retained as a belt-and-suspenders measure
(`gc.collect()` + `torch.cuda.empty_cache()`), but the CPU-reload step that upstream
used after the direction analyzer has been removed.

**Status:** design decision documented for future readers on discrete-GPU hardware
who may wonder why this branch diverges from upstream's memory management.
