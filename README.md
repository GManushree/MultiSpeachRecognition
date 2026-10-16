# GPU-resident multi-speaker ASR

A working, tested research implementation of a multi-speaker speech recognition pipeline:
log-Mel features → self-tuning diarization with encoder-decoder attractors → attractor-
conditioned speech separation → CTC recognition per speaker → N-best language-model
rescoring → chronological speaker-attributed transcript. The pipeline is built to keep all
acoustic tensors on the device, with **one host→device and one device→host transfer per
processing block**, instrumented and verified.

> **Read this first.** Development hardware had **no GPU** (2 CPU cores, 5.8 GiB memory limit)
> and no access to AMI, CHiME-7 or pretrained models. Everything was trained from scratch on
> a synthetic text-to-speech corpus and measured on CPU. The GPU code paths exist and are
> tested where possible, but **no GPU speed-up has been measured**, and the system's
> recognition accuracy is low (test cpWER 64.5 %). See `docs/limitations.md`.

## Results at a glance (CPU, synthetic test set, 6 unseen speakers)

| | Value |
|---|---|
| cpWER / DER / speaker-count accuracy | 64.5 % / 19.5 % / 86.5 % |
| SI-SDRi (2 / 3 speakers) | 9.2 / 7.4 dB |
| Host↔device crossings per block (resident vs cascaded) | **2** vs 20–25 |
| 500 ms block latency, resident pipeline | 41.8 ms median, RTF 0.084 (target < 0.2 met on CPU) |
| Same, cascaded baseline / resident with dynamic K | 29.3 ms / 24.6 ms (faster on CPU, see below) |
| Joint vs separately trained | cpWER 64.5 vs 66.1 %, SI-SDRi(K=2) 9.2 vs 8.9 dB, DER 19.5 vs 18.7 % |

Full tables, ablations and figures: [`docs/benchmark_results.md`](docs/benchmark_results.md).

## Architecture

```
audio ─H2D─▶ LogMel ─▶ STDN (4×attn, d256, h4) ─▶ EDA attractors a_k, p_exist(k), p(t,k)
                              │                                │
                              └──────▶ Separator (Conv-TasNet + FiLM(a_k) + activity) ─▶ S waveforms
                                                                                    │
            LogMel ─▶ CTC ASR (CNN + 4×Transformer) ─▶ on-device N-best ─▶ CTC + LM rescoring
                                                                                    │
transcript ◀─ aggregation ◀─D2H─ one packed buffer (activity, p_exist, tokens, speaker ids)
```

| Module | Code | What it does |
|---|---|---|
| 1 Features | `src/audio/mel_features.py`, `src/cuda/` | 80 log-Mel, 10 ms hop, batched STFT; fused C++/CUDA power→sparse-Mel→log kernel |
| 2A STDN | `src/models/stdn.py`, `attractors.py` | `p_exist(k)=σ(wᵀa_k)`, `K̂=count(p_exist>0.5)`, `p(t,k)=σ(z_tᵀa_k)`; overlap allowed |
| 2B Separator | `src/models/separator.py` | one waveform per attractor; FiLM scaling/shifting; PIT training |
| 3 ASR | `src/models/asr.py` | character CTC; greedy, on-device N-best, host prefix beam search |
| 4 Rescoring | `src/models/rescoring.py` | exact on-device CTC score + char-LSTM LM; serial / CUDA streams / batched |
| 5 Aggregation | `src/models/aggregation.py` | validation, de-dup, merge, sort; overlapping speakers kept; JSON/TXT/SRT |
| Joint loss | `src/models/joint_model.py` | `L_total = L_sep + λ1·L_diar + λ2·L_asr`, one shared Hungarian assignment |

Details: [`docs/architecture.md`](docs/architecture.md), [`docs/implementation_details.md`](docs/implementation_details.md).

## GPU residency and CUDA

* `ResidentPipeline` moves the audio block to the device once and returns one packed buffer.
  Speaker-count decisions, decoding, rescoring and streaming speaker tracking all stay on the
  device, so the default path has no host synchronisation inside the block.
* Two library calls that would add hidden host syncs were designed around: packed LSTM
  sequences (need host lengths) and `F.ctc_loss` on CUDA (copies lengths to the host); the
  latter is replaced by an on-device CTC forward pass that matches it exactly.
* `TransferTracker` counts every crossing and its bytes; on CUDA a profiler check asserts
  exactly one HtoD and one DtoH memcpy per block (test skipped here: no GPU).
* `CascadedPipeline` is the explicit conventional baseline with identical weights. A
  controlled 4-crossing variant supports the `L_sync = (T₄ − T₂)/2` estimate.
* Custom kernel: `src/cuda/kernels.cu` (CUDA, **not compiled here**) and the identical CPU
  kernel in `src/cuda/bindings.cpp` (built and validated, max error 1.5e-5 vs the float64
  reference). Build: `python -m src.cuda.build`.

**Finding on CPU:** fixed-capacity residency (always processing all 3 speaker streams so that
`K̂` never has to be read on the host) costs more compute than it saves when crossings are
free. The dynamic-K resident variant (one sync) was fastest. Whether the trade-off flips on a
GPU is the experiment this repository is ready to run but could not run.

## Quick start

```bash
pip install -r requirements.txt
python scripts/check_environment.py                         # environment + compatibility report
python scripts/prepare_data.py --config configs/training.yaml   # synthetic corpus (~3.5 min)
python scripts/train.py --config configs/training.yaml      # all stages (≈2.7 h on 2 CPU cores)
python scripts/evaluate.py --config configs/evaluation.yaml             # test set
python scripts/evaluate.py --config configs/evaluation.yaml --ablations # ablations 1, 2, 5, 6
python scripts/benchmark.py --config configs/benchmark.yaml             # latency/throughput/memory
python scripts/transcribe.py --audio data/example.wav --checkpoint checkpoints/joint.pt \
       --output results/transcript.json --srt results/transcript.srt   # add --streaming for blocks
uvicorn src.api.main:app --host 127.0.0.1 --port 8000      # POST /transcribe, GET /health, /model-info
python -m pytest                                            # 87 tests (5 need CUDA)
python scripts/plot_results.py                             # figures in results/figures/
```

Training runs in stages (LM → diarization → ASR → separation → ASR adaptation → joint); see
[`docs/training_guide.md`](docs/training_guide.md), including AMI and CHiME-7 preparation and
the documented deviation from `L_total` (an auxiliary clean-reference CTC term, off by default).
Evaluation conventions (cpWER, DER collar and overlap handling, SI-SDRi with PIT, latency
definitions): [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

Hardware: runs on CPU (as developed). On a CUDA machine set `precision: fp16` in the configs;
the CUDA kernel is JIT-compiled automatically when `nvcc` is present.

## Implementation status

| Component | Implementation status | Tests executed | Results | Remaining limitations |
|---|---|---|---|---|
| Environment report | Implemented and tested | script run | `results/environment.json` | – |
| Audio loading, resampling, blocks | Implemented and tested | 5 | pass | multichannel is downmixed |
| Log-Mel features (torch) | Implemented and tested | 6 | 1.5e-5 max error vs float64 reference | – |
| Fused feature kernel, C++ CPU | Implemented and tested | 4 | matches reference and dense path | no speed claim |
| Fused feature kernel, CUDA | Blocked by environment or data | 1 (skipped) | – | never compiled (no nvcc/GPU) |
| STDN / EDA diarization | Implemented and tested | 7 incl. toy overfit, padding invariance | test DER 19.5 %, count acc 86.5 % | K=3 under-counted (55 % acc) |
| Attractor-conditioned separator | Implemented and tested | 5 incl. toy overfit (>10 dB) | SI-SDRi 9.2 / 7.4 dB (K=2/3) | −5.5 dB on single-speaker input |
| CTC ASR + decoders | Implemented and tested | 9 incl. toy overfit, bf16 | clean WER 12.6 % (ASR stage) | positional bias; 44.8 % oracle WER after adaptation |
| Pretrained ASR baseline (wav2vec2) | Implemented but not fully validated | none | – | no model download possible |
| LM rescoring, serial / batched | Implemented and tested | 5 | identical selections; batched 8.4 vs serial 15.5 ms | in-domain LM (perplexity 1.34) |
| Rescoring on CUDA streams | Implemented but not fully validated | 1 (skipped) | – | needs a GPU |
| Aggregation and exports | Implemented and tested | 7 | overlap preserved, JSON/TXT/SRT | timing within a region interpolated |
| Joint loss and staged/joint training | Implemented and tested | 7 + full CPU training run | joint beats separate on cpWER/SI-SDRi | small budget; ablation confounded by extra steps |
| Synthetic corpus | Implemented and tested | 7 | 3.5 h train, speaker-disjoint splits | TTS speech only |
| AMI adapter | Implemented but not fully validated | 1 fixture test | – | corpus not available |
| CHiME-7 DASR adapter | Implemented but not fully validated | 1 fixture test | – | data not available; tcpWER not bundled |
| Resident pipeline (2 crossings) | Implemented and tested | 8 | 2 crossings, 0 syncs; RTF 0.084 (CPU) | GPU memcpy count unverified |
| Cascaded baseline | Implemented and tested | 2 | same transcripts as resident+beam | – |
| Streaming inference | Implemented and tested | 2 | 2 crossings/block, consistent timestamps | accuracy not evaluated on test set |
| FastAPI service | Implemented and tested | 3 + live server run | /health, /model-info, /transcribe | single shared model, one request at a time |
| Evaluation (WER, cpWER, DER, SI-SDR) | Implemented and tested | 7 | matches jiwer and pyannote | – |
| Latency / throughput / memory benchmarks | Implemented and tested | benchmark run | `results/benchmark/` | CPU only |
| L_sync estimate, crossing probe | Blocked by environment or data | – | CPU value is noise (labelled) | needs a GPU |
| FP16 mixed precision | Blocked by environment or data | bf16 proxy run | bf16: same accuracy, slower on CPU | FP16 needs CUDA |
| AMI / CHiME-7 evaluation | Blocked by environment or data | – | – | datasets not downloadable |

Not created: `notebooks/exploratory_analysis.ipynb` (would have been an empty placeholder;
`scripts/plot_results.py` produces the figures instead).

## Repository layout

```
configs/      model, training, inference, evaluation, benchmark YAML
src/audio     loading, blocks, log-Mel (reference + torch + fused-kernel wrapper)
src/cuda      kernels.cu, bindings.cpp (CPU kernel + bindings), build.py
src/models    stdn, attractors, separator, asr, rescoring, aggregation, joint_model
src/data      eSpeak TTS wrapper, sentence generator, synthetic corpus, AMI, CHiME-7
src/training  datasets, losses, checkpoints, trainer loop, stages
src/inference pipeline (resident/cascaded), streaming, offline
src/evaluation wer, der, separation_metrics, latency, throughput, transfer_profiler, evaluate
src/api       FastAPI app and schemas
scripts/      check_environment, prepare_data, train, evaluate, benchmark, transcribe, plot_results
tests/        87 tests (82 pass on CPU; 5 require CUDA and are skipped)
docs/         architecture, implementation, training, evaluation protocol, results, limitations
```
