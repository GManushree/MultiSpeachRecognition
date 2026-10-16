# Benchmark and evaluation results

**Every number here was measured on CPU** (x86_64, 2 threads, PyTorch 2.14.1, no GPU, Python
3.13) with checkpoint `checkpoints/joint.pt` (joint stage, step 400) unless stated. Accuracy is
on the **synthetic eSpeak test set** (200 mixtures, 6 speakers unseen in training). Nothing was
measured on AMI or CHiME-7. Raw outputs: `results/eval/eval_test.json`,
`results/eval/ablations.json`, `results/benchmark/benchmark.json`,
`results/benchmark/latency_500ms.csv`, figures in `results/figures/`.

## 1. System accuracy (test set, offline, resident pipeline)

| Metric | Value |
|---|---|
| cpWER | **64.5 %** (S 983, D 300, I 62, N 2085) |
| DER (collar 0.25 s per side, overlap scored) | **19.5 %** (miss 7.9, FA 4.7, confusion 6.9) |
| Speaker-count accuracy | 86.5 % (mean error −0.085) |
| SI-SDRi, K = 2 / K = 3 | **9.2 dB / 7.4 dB** |
| SI-SDRi, K = 1 | −5.5 dB (the separator degrades single-speaker input) |
| References without an estimated stream | 22 of 368 |
| Oracle-source WER (ASR on clean references) | 44.8 % |

The recogniser is the weakest component. The ASR stage alone reached 12.6 % greedy WER on
clean test sources; adapting it to separator outputs raised that to ~45–50 % (see §4 and
`docs/limitations.md`). cpWER of 64.5 % is far from usable transcription; the system is a
working, measurable implementation, not a trained production model.

## 2. Latency, 500 ms blocks (20 items × 3 repeats after warm-up)

| Variant | Crossings / block | Explicit syncs | Median ms | p95 ms | RTF | Dominant module |
|---|---|---|---|---|---|---|
| resident (paper design, fixed capacity) | **2** | 0 | 41.8 | 51.2 | 0.084 | separation |
| resident, controlled extra round trip | 4 | 0 | 40.7 | 54.1 | 0.082 | separation |
| resident + host beam search | 4 | 0 | 58.7 | 70.7 | 0.118 | rescoring |
| resident, dynamic K (reads K_hat on host) | 3 | 1 | **24.6** | 46.1 | **0.049** | separation |
| cascaded baseline | 20 (median) | 1 | 29.3 | 74.2 | 0.059 | separation |

Per-module medians, resident: features 0.3 ms, diarization 4.2, separation 22.0, ASR 5.2,
rescoring 9.1. Payload per block: resident 32 000 B H2D + 820 B D2H; cascaded 153 kB + 90 kB.

* **RTF < 0.2 is met on this CPU** for every variant (resident 0.084), i.e. under the 100 ms
  budget for a 500 ms block. This is with small models (8.2 M parameters in total).
* **On CPU the GPU-resident design is not faster.** Crossings cost nothing without device
  memory, while `fixed_capacity` always separates and recognises all 3 capacity streams to
  avoid a host sync on `K_hat`. Blocks of 0.5 s usually contain 0–1 speakers, so the cascaded
  baseline (which processes only detected speakers) does less work. The resident pipeline with
  dynamic K, one sync, is fastest. Whether avoiding that sync is worth the extra compute on a
  GPU is the open question this hardware cannot answer.
* **L_sync could not be measured.** `(T_n4 − T_n2) / 2 = −0.53 ms`, i.e. noise: on CPU the
  extra "crossing" is a host memcpy. The benchmark marks this value invalid.

![per-module latency](../results/figures/latency_breakdown.png)

## 3. Block size, throughput, memory, rescoring

| Block | Median ms | RTF median | RTF p95 |
|---|---|---|---|
| 0.5 s | 43.5 | 0.087 | 0.107 |
| 1.0 s | 72.6 | 0.073 | 0.096 |
| 2.0 s | 178.6 | 0.089 | 0.106 |
| 4.0 s | 356.5 | 0.089 | 0.122 |

* Throughput (whole mixtures, single pass): 10.5 audio-s per wall-s overall; 12.7 (K = 1),
  9.5 (K = 2), 9.3 (K = 3) on the fixed-K sets. CPU decode + resample + blocking: 0.13 ms per
  audio second.
* Parameters: STDN 17.1 MB, separator 6.5 MB, ASR 7.9 MB, LM 3.3 MB (FP32). Process RSS
  during the benchmark: 840 MB. GPU peak memory: not measurable.
* Rescoring, identical N-best inputs (3 speakers × 8 hypotheses): serial 15.5 ms, batched
  8.4 ms per block, identical selections and scores. CUDA-stream mode: not measurable (falls
  back to serial on CPU). On CPU, batching beats per-speaker execution.

## 4. Ablations (test sets, offline)

**1 – Joint training** (same pipeline settings)

| Model | cpWER | DER | Count acc | SI-SDRi K=2 / K=3 |
|---|---|---|---|---|
| separately trained (`asr_adapt.pt`) | 66.1 % | **18.7 %** | **89.5 %** | 8.9 / 7.1 dB |
| joint fine-tuning (`joint.pt`) | **64.5 %** | 19.5 % | 86.5 % | **9.2 / 7.4 dB** |
| first joint recipe (`joint_v1_collapsed.pt`) | 85.7 % | 19.7 % | 85.0 % | 8.5 / 7.1 dB |

Joint fine-tuning improved recognition and separation and slightly hurt diarization.
Confound: the joint model got 400 extra optimisation steps. The first recipe (lr 1e-4, no ASR
adaptation stage) drove the recogniser towards blank output.

**2 – Precision** (CPU: bf16 autocast as the only available proxy; FP16 needs CUDA)

| | cpWER | DER | Mean latency per mixture | RTF |
|---|---|---|---|---|
| fp32 | 64.5 % | 19.5 % | 331 ms | 0.078 |
| bf16 | 64.3 % | 19.5 % | 389 ms | 0.093 |

No accuracy change; bf16 is slower on this CPU (no bf16 matrix units). 500 ms block medians:
42.4 ms vs 44.0 ms.

**5 – Overlap** (K = 2, measured overlap ratio = overlapped / total speech)

| Overlap | 0.00 | 0.085 | 0.184 | 0.325 |
|---|---|---|---|---|
| cpWER | 63.4 % | 66.0 % | 68.1 % | 71.7 % |
| DER | 16.2 % | 17.6 % | 21.9 % | 21.0 % |
| SI-SDRi | 16.1 dB | 10.1 dB | 7.3 dB | 6.4 dB |

**6 – Speaker count** (target overlap 0.3; trained and configured capacity: 3)

| K | 1 | 2 | 3 |
|---|---|---|---|
| cpWER | 36.6 % | 68.5 % | 80.4 % |
| DER | 8.1 % | 26.4 % | 35.7 % |
| Count accuracy (mean error) | 93.3 % (+0.07) | 90.0 % (−0.03) | 55.0 % (−0.45) |
| SI-SDRi | −5.0 dB | 6.5 dB | 6.3 dB |

With three speakers the STDN under-counts in almost half the recordings.

**3 – CUDA streams, 4 – GPU residency:** measured parts are in §2–3; the GPU-specific
quantities (stream concurrency, memcpy/sync cost, `L_sync`, isolated vs loaded crossings) are
blocked by the absence of a GPU.

## 5. Analytical latency model (prediction, not measurement)

`RTF ≈ (T_comp + n · L_sync) / T_b` with the measured `T_comp = 41.8 ms`, `T_b = 0.5 s` and
hypothetical `L_sync` values (not measured anywhere in this work):

| L_sync | n = 2 | n = 4 | n = 25 |
|---|---|---|---|
| 10 µs | 0.0836 | 0.0837 | 0.0841 |
| 50 µs | 0.0838 | 0.0840 | 0.0861 |
| 200 µs | 0.0844 | 0.0852 | 0.0936 |

At this model size the crossing term is predicted to be small next to compute unless `L_sync`
is in the hundreds of microseconds and the crossing count is large.

## 6. Training (CPU, ≈ 2.7 h of optimisation for the final checkpoint chain)

| Stage | Steps | Wall time | Validation result |
|---|---|---|---|
| LM | 534 | 1.6 min | 0.293 nats/char (perplexity 1.34) |
| diarization | 2500 | 24 min | frame DER 0.200, count acc 0.867 |
| ASR (clean) | 3000 | 38 min | greedy WER 14.8 % |
| separation (3 s crops) | 2100 | 59 min (two runs, resumed) | pooled SI-SDRi 5.17 dB |
| ASR adaptation | 1000 | 33 min | separated-stream WER 69.0 % |
| joint | 400 | 7 min | separated-stream WER 66.9 %, SI-SDRi 5.52 dB, count acc 0.893 |

![training curves](../results/figures/training_curves.png)
