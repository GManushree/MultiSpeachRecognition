# Training guide

## 1. Data

```bash
python scripts/prepare_data.py --config configs/training.yaml            # synthetic corpus (local, ~3.5 min CPU)
python scripts/prepare_data.py --config configs/training.yaml --only ami data.ami.root=/path/to/ami
python scripts/prepare_data.py --config configs/training.yaml --only chime7 data.chime7.root=/path/to/chime7 data.chime7.split=dev
```

**Synthetic corpus** (`src/data/synthetic.py`): eSpeak NG voices define 36 speakers (24 train /
6 val / 6 test, disjoint). Each mixture has 0–3 speakers (5/25/45/25 %), one sentence each,
relative gains ±5 dB, target overlap 0–60 %, white noise at 30 dB SNR. References: scaled
source waveforms, transcripts, VAD segments computed on the sources. Ablation sets with fixed
K and overlap are generated from the test speakers. Generation is deterministic given the seed
(tested).

**AMI** (`src/data/ami.py`): download the corpus and NXT manual annotations yourself (CC BY 4.0;
accept the terms at the AMI site). Expected layout is documented in the module docstring.
Splits follow the full-corpus-ASR partition. Windows of 8 s; utterances cut by a window keep
their speaker activity but no transcript.

**CHiME-7 DASR** (`src/data/chime7.py`): prepare CHiME-6 / DiPCo / Mixer 6 with the official
`chime7_dasr` recipe, then point `data.chime7.root` at it. The official metric is tcpWER via
`meeteval`, which is not bundled here; this adapter produces manifests and reference segments.

Neither real corpus was available in the development environment; both adapters are covered
only by unit tests on small fixtures.

## 2. Staged training then joint fine-tuning

```bash
python scripts/train.py --config configs/training.yaml                   # all stages in order
python scripts/train.py --stage diarization,asr --threads 2              # a subset
python scripts/train.py --resume                                         # continue from *_last.pt
python scripts/train.py --smoke                                          # pipeline check, few steps
python scripts/train.py stages.joint.steps=2000 weights.lambda_asr=0.5   # overrides
```

| Stage | Trains | Loss | Frozen | Init |
|---|---|---|---|---|
| `lm` | char LSTM LM | NLL on training transcripts | – | random |
| `diarization` | STDN | `L_diar` | separator, ASR | random |
| `asr` | ASR | CTC on clean sources (+ gain, 20–40 dB noise) | STDN, separator | `diarization.pt` |
| `separation` | separator | `L_sep` on 3 s crops, attractors from the frozen STDN | STDN, ASR | `asr.pt` |
| `asr_adapt` | ASR | CTC on separator outputs + clean-reference CTC | STDN, separator | `separation.pt` |
| `joint` | all | `L_sep + λ1 L_diar + λ2 L_asr` (+ clean-reference CTC) | – | `asr_adapt.pt` |

Frozen modules are put in eval mode and get no gradient (tested). Until `joint`, every stage
updates one module with that module's loss; only `joint` optimises all three terms together,
with gradients from `L_asr` reaching the separator and the STDN. `asr_adapt.pt` is therefore
the "separately trained" control for ablation 1.

Deviation from the paper's objective: `asr_adapt` and `joint` add `lambda_asr_clean · CTC` on
the clean references placed at their mixture offsets (`weights.lambda_asr_clean`, 0 by
default). Without it the recogniser forgot clean speech (12.6 % → 51.6 % WER) when adapted to
separator outputs. Setting it to 0 recovers the exact `L_total`.

Recipe history (all checkpoints and logs kept): a first joint run directly after separation
at lr 1e-4 degraded the recogniser (`joint_v1_collapsed.pt`); a second recipe without the
clean term is in `checkpoints/v2_no_clean/` (test cpWER 67.5 %); the final recipe reaches 64.5 %.

Memory: full-length separation batches were killed by the 5.8 GiB memory limit, hence 3 s
crops (speakers silent in a crop are dropped from its references) and batch size 1 for joint.

A missing initialisation checkpoint raises an error; nothing silently falls back to random
weights.

## 3. Hardware

The configuration in `configs/training.yaml` was sized for the 2-core CPU development machine
(see `docs/benchmark_results.md` for the actual run). On a GPU set `precision: fp16` and
increase batch sizes and steps; `device: auto` selects CUDA when available. Fine-tuning from a
pretrained recognizer requires a checkpoint compatible with `CTCASR`; the optional wav2vec2
baseline in `src/models/asr.py` is for comparison only.
