# Architecture

This document describes what the code in `src/` implements. Paths are given for every component.

## Data flow per processing block (GPU-resident mode)

```
host                          device (GPU when available; CPU fallback runs the same code)
────                          ──────────────────────────────────────────────────────────────
audio block (float32) ──H2D──▶ LogMel (Module 1)                       (B, 80, T)   10 ms hop
  crossing 1                    │
                                ▼
                              STDN (Module 2A)  conv x4 ─ 4 x self-attention (d=256, h=4)
                                │  z_t (T', 256) ── EDA (LSTM enc/dec) ── a_k, p_exist(k)
                                │  p(t,k) = sigmoid(z_t · a_k)          T' = 40 ms frames
                                ▼
                              Separator (Module 2B)  Conv-TasNet encoder ─ TCN with FiLM(a_k)
                                │  + activity channel p(t,k) ─ mask_k ─ decoder   (S, N) waveforms
                                ▼
                              LogMel (Module 1, reused) ─ ASR (Module 3)  CNN + 4 x Transformer
                                │  CTC log-probs (S, T'', 29)           T'' = 20 ms frames
                                ▼
                              on-device N-best (2^m alternative alignments)
                              Rescoring (Module 4): exact CTC forward + char-LSTM LM, argmax
                                │
                              pack: p_exist, activity, best tokens, emission frames, [speaker ids]
segments ◀────────────D2H──────┘
  crossing 2
Aggregation (Module 5, host): words placed on each speaker's active regions → segments → sorted transcript
```

`S` is the configured speaker capacity (3). In the default `fixed_capacity` mode all S
streams are separated and recognised and absent speakers are masked out on the host after
the single D2H copy; this avoids reading the data-dependent speaker count `K_hat` on the host
in the middle of the block, which would add a crossing and a blocking synchronisation.

## Modules

| Module | File | Notes |
|---|---|---|
| 1 Log-Mel features | `src/audio/mel_features.py`, `src/audio/cuda_features.py`, `src/cuda/*` | 16 kHz, 25 ms Hann, 10 ms hop, 80 HTK Mel bands, `log(mel + 1e-6)`. Batched `torch.stft`. Fused power→sparse-Mel→log kernel in C++ (CPU, built and tested) and CUDA (written, not compiled here). |
| 2A STDN | `src/models/stdn.py`, `src/models/attractors.py` | EEND-EDA style. 4.47 M parameters. `K_hat = count(p_exist > 0.5)`, independent sigmoids per (t, k) so overlap is representable. |
| 2B Separator | `src/models/separator.py` | Encoder L=32/stride 16, 8 FiLM-TCN blocks (2 × 4, dilations 1…8), bottleneck 128, hidden 256. 1.70 M parameters. Stream k is defined by attractor k. |
| 3 ASR | `src/models/asr.py` | Character CTC (29 symbols). Conv2d front-end (×2 time, ×4 freq), 4 Transformer layers d=192. 2.07 M parameters. Decoders: greedy, on-device N-best, host prefix beam search, Viterbi forced alignment. |
| 4 Rescoring | `src/models/rescoring.py` | Acoustic score = exact CTC log-likelihood (on-device forward algorithm), LM = 2-layer char LSTM (0.8 M parameters). Modes: serial, CUDA streams, batched. |
| 5 Aggregation | `src/models/aggregation.py`, `src/inference/pipeline.py` | Word times from the speaker's diarized activity (default) or CTC emission frames; validation, de-duplication, same-speaker merging, chronological sort; JSON / text / SRT export. Overlapping speakers are kept overlapping. |
| Joint model | `src/models/joint_model.py`, `src/training/losses.py` | `L_total = L_sep + λ1 L_diar + λ2 L_asr` with one Hungarian assignment shared by all three terms. |

## Pipelines

* `ResidentPipeline` (`src/inference/pipeline.py`): the design above. Options:
  `fixed_capacity=False` (+1 D2H for the speaker mask, +1 host sync), `decoder="beam"`
  (logits D2H, hypotheses H2D), `n_crossings=4` (identical compute plus a forced round trip of
  the separated waveforms, used to estimate `L_sync`).
* `CascadedPipeline`: the conventional baseline with identical weights. Each module boundary
  goes through host memory, `K_hat` is decided on the host, each speaker is recognised in a
  separate call, beam search and forced alignment run on the host, and rescoring inputs are
  uploaded again. With three speakers this is 25 crossings per block.
* `StreamingTranscriber` (`src/inference/streaming.py`): blocks with left context; the speaker
  bank lives on the device and is matched inside the same block computation, so streaming
  keeps 2 crossings per block.

## Transfer accounting

Every crossing goes through `TransferTracker.h2d/d2h` (`src/evaluation/transfer_profiler.py`),
which records direction, bytes and purpose. On CUDA `profile_memcpy` additionally counts the
memcpy activities seen by the profiler (`tests/test_cuda.py::test_cuda_resident_pipeline_memcpy_count`
asserts exactly one HtoD and one DtoH). Two library behaviours that would add hidden syncs were
designed around:

* `nn.utils.rnn.pack_padded_sequence` needs host-side lengths, so the EDA only packs when
  lengths are given (training); inference blocks are unpadded and never pack.
* `F.ctc_loss` on CUDA copies its length tensors to the host. Rescoring therefore uses
  `ctc_log_likelihood`, an on-device CTC forward pass that matches `F.ctc_loss` exactly (tested).
