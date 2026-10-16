# Evaluation protocol

## Manifest format (JSON lines)
```json
{"id": "test_00001", "split": "test", "mix_path": "test/test_00001/mix.wav", "duration": 4.21,
 "sample_rate": 16000, "n_speakers": 2, "target_overlap": 0.31, "overlap_ratio": 0.18,
 "speakers": [{"speaker_id": "tts031", "text": "...", "offset": 0.2, "duration": 1.9,
               "segments": [[0.2, 2.1]], "path": "test/test_00001/s0.wav"}, ...]}
```
`path` (reference source) exists only for the synthetic corpus; AMI/CHiME manifests set
`has_sources: false` and separation metrics are skipped for them.

## Recognition
* **cpWER** (concatenated minimum-permutation WER, CHiME-6 style): per recording, each
  reference speaker's words are concatenated, likewise for each hypothesis speaker; the
  mapping minimising total errors is found with the Hungarian algorithm; unmatched reference
  speakers count as deletions and unmatched hypothesis speakers as insertions.
  `WER = (S + D + I) / N`, accumulated over the corpus (counts, not averaged rates).
* **Oracle-source WER**: greedy ASR on the clean reference sources; isolates the recogniser.
* Text normalisation: lower case, letters, apostrophe and space only.
* Our WER implementation is cross-checked against `jiwer` in `tests/test_metrics.py`.

## Diarization
* **DER** from `pyannote.metrics.DiarizationErrorRate`.
* Reference: VAD segments of each source (synthetic) or annotation segments (AMI/CHiME).
* Hypothesis: STDN activity `p(t,k) > 0.5` for present speakers, 40 ms frames, segments
  shorter than 80 ms removed.
* Collar: **0.25 s per boundary side** (md-eval convention). pyannote's argument is the total
  width, so 0.5 is passed (`src/evaluation/der.py`; behaviour checked in tests).
* Overlapped speech is scored. Speaker mapping: optimal one-to-one. UEM: the whole recording.
* Speaker-count accuracy: `K_hat == K`.

## Separation
* **SI-SDR** and **SI-SDRi** (improvement over the unprocessed mixture), zero-mean, with
  permutation-invariant matching (Hungarian on pairwise SI-SDR).
* Streams come from the *estimated* attractors and speaker count; references without an
  estimate are reported as `missed_refs` and are not given a score.

## Latency, throughput, memory
* Block latency: wall time of `process_block`, from the H2D copy to the D2H copy returning
  (the D2H is blocking, so all device work is included). Median, p95 over
  `repeats × n_items` blocks after warm-up. RTF = latency / block duration.
* Per-module time: CUDA events (no extra syncs) on GPU, `perf_counter` on CPU.
* Throughput: audio seconds per wall second over whole mixtures.
* Memory: parameter bytes per module, process RSS, CUDA peak allocation (GPU only).
* Transfers: counts, bytes and labels from `TransferTracker`; CUDA memcpy counts from the
  profiler. Transfer bandwidth time and synchronisation wait are reported separately.

## Analytical latency model
`T_xfer = Σ_i (S_i / B_PCIe + L_sync)`, `RTF ≈ (T_comp + n · L_sync) / T_b`, and
`L_sync ≈ (T_{n=4} − T_{n=2}) / 2` from the controlled 4-crossing variant, which performs
identical compute plus one extra round trip of the separated waveforms. Assumptions: same
compute in both runs, the extra payload's bandwidth time is negligible or subtracted, no
overlap of the extra copy with compute. Model outputs are always labelled as predictions.
