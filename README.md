<h1 align="center">AMID</h1>

<p align="center">
  <strong>Explanation and Temporal Alignment with Agent Routed Mixture-of-Experts<br>for Multimodal Sentiment Analysis</strong>
</p>

<p align="center">
  <a href="#abstract"><img src="https://img.shields.io/badge/Paper-Manuscript-3974AD?style=flat-square" alt="Paper: manuscript"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/Python-3.9%2B-77A53B?style=flat-square" alt="Python: 3.9 or newer"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/Framework-PyTorch-DF6545?style=flat-square" alt="Framework: PyTorch"></a>
  <a href="https://github.com/liangyubuaa/AMID"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square" alt="Code: GitHub"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> &middot;
  <a href="#key-contributions">Key Contributions</a> &middot;
  <a href="#results">Results</a> &middot;
  <a href="#getting-started">Getting Started</a> &middot;
  <a href="#parameters">Parameters</a> &middot;
  <a href="#reference">Reference</a>
</p>

**AMID** combines evidence from modality-specific agents, temporal alignment, and adaptive expert fusion to predict sentiment from the language, audio, and visual modalities.

| Explanation-driven representations | Fine-grained temporal alignment | Sample-specific fusion |
| :--- | :--- | :--- |
| Make implicit affective cues explicit through structured agent reports. | Connect explanation tokens with the nonverbal events they describe. | Adapt modality contributions while retaining evidence from every stream. |

![AMID architecture: Explanation Agents, Analysis Agent, Explanation Attention, EATS, and Agent Routed MOE](./assets/amid_architecture.png)

*Framework overview. Explanation Agents describe audio and visual evidence. The Analysis Agent supplies a Final Description and Fusion Weight. Explanation Attention, token reweighting, and EATS refine the representations before expert fusion and sentiment prediction.*

## Abstract

Multimodal sentiment analysis (MSA) aims to infer sentiment polarity and intensity from the language, audio, and visual modalities. However, affective evidence can remain implicit in modality representations, the most informative modality can vary across samples, and explanation tokens and nonverbal sequences have different temporal resolutions. To address these issues, we propose AMID, a multi-agent framework that integrates representation learning driven by explanations, cross-modal temporal synchronization, and adaptive expert fusion. AMID first uses modality-specific Explanation Agents to convert audio and sampled video into structured evidence reports. Explanation Attention encodes these reports and connects their cue tokens with modality features, while a tri-modal Analysis Agent generates a Final Description and a sample-specific Fusion Weight. The Final Description reweights salient audio and visual tokens, and the Explanation Attention Temporal Synchronizer derives temporal anchors and reliability scores from explanation attention to synchronize the two nonverbal streams. Finally, Agent Routed Mixture-of-Experts fuses language, audio, and visual representations while retaining every modality above a routing floor. Experiments on CMU-MOSI, CMU-MOSEI, and CH-SIMS show that AMID achieves state of the art performance, with substantial leads in fine grained classification and regression while generalizing well.

## Key Contributions

- We propose AMID, a multi-agent framework for multimodal sentiment analysis that uses explanation features derived from multimodal large language models, together with sample-specific salient affective cues and dominant-modality estimates.
- We introduce modality-specific Explanation Agents to extract audio and visual evidence, yielding structured, comprehensive, and high-level explanation features that expose implicit affective information.
- We design an Analysis Agent to derive emotion-relevant cues and modality weights from a global multimodal view, preserving salient features and determining a more appropriate dominant modality for each sample.
- We develop EATS, an Explanation Attention Temporal Synchronizer that exploits synchronized audio-video timing and explanation attention maps to enable fine-grained alignment of high-level evidence token features.
- Across three public benchmarks, AMID ranks first among the compared systems in all reported metric columns, covering binary and fine-grained classification as well as sentiment-intensity regression.

## Results

Results on CMU-MOSI, CMU-MOSEI, and CH-SIMS, as reported in the paper.

### Main Results

**CMU-MOSI and CMU-MOSEI**

| Dataset | F1 | Acc-2 | Acc-5 | Acc-7 | MAE | Corr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CMU-MOSI | **87.69 / 89.77** | **87.76 / 89.79** | **56.56** | **50.73** | **0.595** | **0.867** |
| CMU-MOSEI | **86.62 / 87.03** | **86.74 / 87.31** | **59.13** | **57.18** | **0.489** | **0.813** |

**CH-SIMS**

| Dataset | F1 | Acc-2 | Acc-3 | Acc-5 | MAE | Corr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CH-SIMS | **87.66** | **87.31** | **80.09** | **60.39** | **0.255** | **0.845** |

**Metric conventions.** F1 is weighted F1. Classification metrics are percentages; MAE and Pearson correlation retain their original scales. Higher is better except for MAE. For MOSI and MOSEI, paired F1 and Acc-2 entries report **Has0 / Non0**: negative versus non-negative with zero retained, followed by negative versus positive with zero removed. CH-SIMS reports the negative versus non-negative setting.

## Getting Started

### Environment

The entry point requires **Python 3.9 or newer** because it uses `argparse.BooleanOptionalAction`. Its main dependencies are PyTorch, Transformers, NumPy, scikit-learn, and tqdm. Install a PyTorch build compatible with your hardware, then install the remaining packages:

```bash
python -m pip install transformers numpy scikit-learn tqdm
```

The script supports CPU, CUDA, and Ascend NPU devices. NPU execution additionally requires a compatible `torch_npu` installation and Ascend runtime. With `--device auto`, the script selects an available NPU first, then CUDA, then CPU.

### Prepare the Inputs

Training uses precomputed modality features and agent outputs.

Prepare a local BERT checkpoint directory and a PKL containing `train`, `valid`, and `test` splits. Each split must provide:

| Field | Shape per split | Content |
| :--- | :--- | :--- |
| `text` | `(N, T_l, D_l)` | Language modality features. |
| `audio` | `(N, T_a, D_a)` | Audio modality features. |
| `vision` | `(N, T_v, D_v)` | Visual modality features. |
| `text_clue_bert` | `(N, 3, L_l)` | BERT inputs for the language cue sequence. |
| `audio_clue_bert` | `(N, 3, L_a)` | BERT inputs for the audio explanation. |
| `visual_clue_bert` | `(N, 3, L_v)` | BERT inputs for the visual explanation. |
| `final_description_bert` | `(N, 3, L_f)` | BERT inputs for the Final Description. |
| `fusion_weights` | `(N, 3)` | Nonnegative weights in language, audio, visual order, summing to one. |
| `regression_labels` | One scalar per sample | Sentiment regression targets. |

Here `N` is the split size, `T` denotes feature sequence length, `D` denotes feature dimension, and `L` denotes token length. The three channels of each BERT input are `input_ids`, `attention_mask`, and `token_type_ids`, in that order. Samples must correspond across all fields. The loader casts modality features to FP32 and BERT inputs to integer tensors.

The model loads BERT with `local_files_only=True` and keeps it frozen in the default training path. Its explanation projections expect 768-dimensional BERT hidden states. For MOSEI, prepare a local `bert-base-uncased` checkpoint and inputs tokenized with its matching tokenizer.

### Train on MOSEI

From the repository root, replace the example paths with your prepared inputs. Keep data, checkpoints, and training output outside the source directory:

```bash
python train.py --pkl "../data/mosei.pkl" --bert_dir "../checkpoints/bert-base-uncased" --output_dir "../runs" --run_name amid_mosei --device auto
```

The script evaluates the validation and test sets each epoch. By default, it selects the checkpoint using the **lowest validation loss** and evaluates EMA weights from epoch 4 onward. Each run writes the following files beneath the explicitly selected output directory:

| Output | Content |
| :--- | :--- |
| `best.pt` | Selected model state, arguments, epoch, and metric record. |
| `train_log.jsonl` | Per-epoch training loss, validation metrics, and test metrics. |
| `status.json` | Latest epoch, best epoch, and current metric record. |

Classification metrics in the training logs use fractions, while the paper tables above display percentages.

## Parameters

Defaults below are taken from [`train.py`](./train.py) and its arguments passed into [`ModelA`](./model/model.py).

| Setting | Argument | Default |
| :--- | :--- | :--- |
| Training epochs | `--epochs` | `50` |
| Training batch size | `--batch_size` | `8` |
| Evaluation batch size | `--eval_batch_size` | `16` |
| Optimizer | Defined in the training loop | AdamW |
| Learning rate | `--lr` | `5e-5` |
| Weight decay | `--weight_decay` | `1e-4` |
| Hidden dimension | `--hidden_dim` | `160` |
| Attention heads | `--heads` | `4` |
| Feature encoder layers | `--feature_layers` | `2` |
| Explanation alignment layers | `--align_layers` | `2` |
| Dropout | `--dropout` | `0.15` |
| FD reweighting position | `--mode_e` | `b` (after alignment) |
| FD gate strength | `--scale_e` | `0.15` |
| Temporal synchronization | `--flag_i`, `--mode_i` | Enabled, `i` (EATS) |
| EATS temporal distance width | `--scale_i` | `0.08` |
| EATS reliability-bias weight | `--gain_i` | `0.20` |
| EATS FD-derived offset strength | `--offset_i` | `0.0` |
| Minimum modality routing weight | `--limit_x` | `0.1` |
| FW bias strength | `--scale_x` | `2.0` |
| Base regression loss | `--loss_type`, `--smooth_l1_beta` | Smooth L1, `0.5` |
| Gradient clipping | `--grad_clip` | `1.0` |
| EMA decay | `--ema_decay` | `0.997` |
| First EMA epoch | `--ema_start_epoch` | `4` |
| Checkpoint selection | `--selection_metric`, `--selection_mode` | Validation `loss`, `min` |
| Early stopping patience | `--early_stop_patience` | `0` (disabled) |

## Reference
