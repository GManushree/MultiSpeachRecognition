# Limitations

Everything below was observed in, or follows from, the development environment:
Ubuntu 24.04, Python 3.13, PyTorch 2.14.1 (CUDA 13.0 wheel), **2 CPU cores, no GPU**, a 5.8 GiB
memory cap for all processes, and network access limited to PyPI/GitHub.

## Hardware: no GPU
* Every latency, throughput and memory number in this repository was measured **on CPU**. The
  CPU path runs the same code as the GPU path, but it is not a GPU-optimised implementation and
  its numbers say nothing about GPU performance.
* Not measured, blocked by hardware: GPU latency/RTF, CUDA memcpy counts from the profiler,
  PCIe synchronisation overhead `L_sync`, isolated vs loaded crossings, FP16 autocast, CUDA
  stream concurrency in rescoring, GPU peak memory, and repeats across GPU architectures.
  The code paths and tests for all of these exist (`tests/test_cuda.py`, `@cuda` tests are
  skipped) and run unchanged on a CUDA machine.
* On CPU a "host↔device crossing" is a counted no-op copy. The 2- vs 4-crossing timing
  difference measured here is therefore *not* an estimate of `L_sync`; the benchmark labels
  it as such.
* The custom CUDA kernel (`src/cuda/kernels.cu`) was written but **never compiled or run**
  (no `nvcc`). Its CPU twin in `bindings.cpp` is built and validated against the reference.
  Whether the fused kernel beats cuBLAS on a GPU is unknown.

## Data
* **AMI and CHiME-7 DASR were not evaluated.** Neither could be downloaded here. Their
  adapters exist and are unit-tested on tiny fixtures only.
* All accuracy numbers come from a **synthetic eSpeak NG corpus** (24 training voices, 6
  unseen test voices, ~3.5 h). TTS speech is far more regular than meeting speech, there is
  no reverberation and only 30 dB white noise. Results do not transfer to real recordings.
* The sentence generator is a small grammar shared by train and test. The character LM
  therefore models the test sentences very well (validation perplexity 1.34), so LM
  rescoring gains here are an upper bound, not a realistic estimate.
* Speaker count per mixture is at most 3 in training; `max_speakers = 3` is the decoding
  capacity. Behaviour with more speakers is untested.

## Models and training
* No pretrained checkpoints were available (Hugging Face unreachable). All models were
  trained from scratch on CPU with small budgets; none can be claimed to have converged.
  The optional wav2vec2 baseline wrapper is untested.
* Compute budget forced compromises, documented in `configs/training.yaml`: separation is
  trained on 3 s crops (full-length batches were killed by the memory limit), joint
  fine-tuning uses batch size 1 and a few hundred steps.
* **Recognition quality is low**: test cpWER 64.5 %. The separator reaches only ~9 dB
  SI-SDRi for two speakers after ~1 h of CPU training, and the recogniser is the weakest link.
* **Positional bias in the recogniser.** It was pretrained on trimmed utterances that always
  start at t = 0. Inside mixtures speech starts later, and the full-attention CTC model then
  (a) recognises worse (clean references at their mixture offset: CTC loss 7.4 vs ~0.5 when
  trimmed) and (b) emits characters at wrong times. Hence word timestamps come from the
  diarization by default, and timing *within* a speaker's region is interpolated, not measured.
  Pretraining on randomly offset/padded utterances would be the proper fix (not done).
* **Forgetting and a deviation from L_total.** Adapting the recogniser to separator outputs
  alone raised clean-speech WER from 12.6 % to 51.6 %. The final recipe adds an auxiliary
  clean-reference CTC term (documented, configurable, off by default). Even so, oracle-source
  WER of the final model is 44.8 %, much worse than the pretrained recogniser's 12.6 %.
* A first joint fine-tuning run (lr 1e-4, directly after separation pretraining) **degraded
  the recogniser** towards blank outputs (test cpWER 85.7 %). It is kept as
  `checkpoints/joint_v1_collapsed.pt` with its log.
* The separator **degrades single-speaker input** (SI-SDRi −5.5 dB for K = 1).
* With three speakers the speaker count is right in only 55 % of recordings (under-counting).
* The joint-training ablation is confounded by extra optimisation: the joint model receives
  additional steps on top of the separately trained control.
* `device_nbest` (the on-device N-best) is an approximation of beam search (alternative
  alignments of the lowest-confidence frames), not an exact prefix beam search.

## Pipeline
* `fixed_capacity` mode separates and recognises all capacity streams every block to avoid a
  host sync on `K_hat`. On CPU this made the resident pipeline *slower* than the cascaded
  baseline (41.8 vs 29.3 ms per 500 ms block); the dynamic-K variant was fastest (24.6 ms).
  The GPU trade-off is unmeasured. The paper's claim that GPU residency reduces latency is
  therefore neither confirmed nor refuted by this work.
* Streaming with 2 s blocks recognises noticeably worse than offline single-pass decoding on
  the example recording; streaming accuracy was not evaluated on the test set.
* Multichannel input is downmixed (or one channel selected); no beamforming or GSS.
* Streaming uses fixed blocks with left context and re-decodes the context; latency includes
  that recomputation. Speaker tracking across blocks is a cosine-similarity attractor bank
  with greedy matching; long-range identity errors are possible and were not measured on
  real meetings.
