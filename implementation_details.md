# Implementation details

## Module 1 — features
* Reference: `reference_log_mel` (NumPy, float64, own framing and FFT). The torch path and the
  fused kernel are validated against it to an absolute log-domain error below 1e-4
  (measured: 1.5e-5).
* The STFT always runs in FP32 (also under autocast); features are differentiable so ASR
  gradients reach the separator in joint training.
* Fused kernel (`src/cuda/bindings.cpp`, `src/cuda/kernels.cu`): reads the complex STFT once,
  accumulates `|X|^2` only over each band's non-zero bins (`band_lo/band_hi` precomputed from
  the filterbank; every band spans fewer than 40 of the 201 bins) and applies the log. It
  avoids materialising the power spectrogram. It is used only under `no_grad` (it has no
  backward). The CUDA version launches on PyTorch's current stream. Whether it is faster than
  the cuBLAS GEMM baseline on a GPU has **not** been measured (no GPU available).

## Module 2A — STDN / EDA
* Masked CMVN, two stride-2 Conv1d layers with explicit zeroing of padded positions between
  them (without this the conv bias leaked padding into valid frames; caught by
  `test_padding_invariance`), pre-norm Transformer encoder with key-padding mask.
* EDA: frames are shuffled during training only. The decoder runs `max_speakers + 1` steps.
* Existence loss follows EEND-EDA: targets `[1]*K + [0]`, positions after `K+1` ignored.
* Speaker count: `K_hat = count(p_exist > threshold)`. A count is flagged *uncertain* if any
  `p_exist` lies within `uncertainty_margin` (0.15) of the threshold; uncertain speakers are
  marked in the transcript rather than dropped.

## Module 2B — separator
* FiLM per TCN block: `h' = (1 + γ(a_k)) h + β(a_k)`, near-identity initialisation (std 1e-2;
  exactly zero would block the gradient into the attractors at initialisation).
* The speaker's activity `p(t,k)` (linearly upsampled) is concatenated to the bottleneck input,
  giving the separator a time-varying cue and a second gradient path into the STDN.
* All K conditioned passes share one encoder pass (K is folded into the batch).

## Module 3 — ASR
* 20 ms output frames. With 40 ms frames the synthetic speech (≈15–20 characters/s) left
  CTC almost no slack, so time subsampling was reduced to ×2.
* Mixed precision via `torch.autocast`; parameters stay FP32 (master weights); CTC and
  log-softmax are computed in FP32; `GradScaler` is used for FP16 training on CUDA.
* `device_nbest`: from per-frame top-2 symbols, the `m` lowest-margin frames are switched to
  their runner-up, giving `2^m` alignments ranked by path score; the top `N` are collapsed
  with a scatter-based CTC collapse. This is an approximation to beam search that keeps
  decoding on the device; `prefix_beam_search` (host) is the exact-prefix alternative.
* `PretrainedCTCBaseline` (wav2vec2 via Transformers) is an optional baseline only. It could
  not be exercised here (no model download possible) and is never substituted silently.

## Module 4 — rescoring
* `score = w_ac · log P_ctc(h|X) + w_lm · log P_lm(h) + w_len · |h|` with `w_ac=1, w_lm=0.3`.
* Streams mode: speaker s runs on `torch.cuda.Stream` `s mod n`. Each side stream waits on an
  event recorded on the main stream after the inputs are produced; inputs are protected with
  `record_stream`; the main stream waits on per-stream completion events. Parameters are only
  read. On CPU the mode falls back to serial (documented, tested).

## Module 5 — aggregation
* Default (`timestamp_source: diarization`): the speaker's active regions (activity > 0.5,
  gaps ≤ 0.3 s merged, regions ≥ 0.1 s) are concatenated and the speaker's words are laid out
  over them in order, proportional to character count. Reason: the CTC emission frames of the
  full-attention recogniser were measured to be off by up to ~1 s (both streams of a test
  mixture emitted the same word at the same frame), while the diarization activity matched the
  references within ~0.05 s. `timestamp_source: ctc` uses emission frame × 20 ms instead.
* Duplicates: same speaker, same text, temporal IoU ≥ 0.5. Merge: same speaker, gap ≤ 0.6 s.

## Joint objective
* Optional auxiliary term `lambda_asr_clean · CTC(ASR(clean reference))` (0 by default; 1 in
  the `asr_adapt` and `joint` stages) - see docs/training_guide.md.
* Per mixture with K references, the first K attractors are matched to references with the
  Hungarian algorithm on `C = w_d · BCE_pair − w_s · SISDR_pair`; this exact PIT solution is
  shared by the activity, separation and ASR losses.
* Non-finite terms are replaced by a graph-preserving zero and counted; the trainer skips
  steps with a non-finite total and aborts after 50.
* Separated streams are RMS-normalised before the ASR front-end (SI-SDR training leaves the
  output scale unconstrained).

## Training system
* AdamW, warm-up + cosine schedule, gradient clipping (5.0), early stopping on the stage's
  validation metric, best/last checkpoints written atomically, `--resume`, crash checkpoint on
  any exception or SIGTERM, JSONL logs with RSS and GPU peak memory.
