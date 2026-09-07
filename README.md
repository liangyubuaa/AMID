# AMID

AMID is a multi-agent framework for multimodal sentiment analysis based on explanation-guided representation learning, fine-grained temporal alignment, and agent-routed adaptive fusion.

![AMID architecture](./assets/amid_architecture.png)

## Abstract

Multimodal sentiment analysis aims to infer sentiment polarity and intensity from the language, audio, and visual modalities, but subtle affective cues, sample-dependent modality importance, and mismatched temporal resolutions make reliable fusion difficult. AMID addresses these challenges with modality-specific Explanation Agents, an Analysis Agent for salient-cue and fusion-weight estimation, EATS for explanation-guided temporal synchronization, and Agent Routed MOE for adaptive multimodal prediction.

## Files

- `train.py`: MOSEI data loading, training loop, validation/test evaluation, and metric logging.
- `model/model.py`: model assembly and forward path. The public class is `ModelA`.
- `model/units.py`: the `UnitA`--`UnitJ` blocks and the final three-path operation.



## Parameters

![Implementation details](./assets/implementation_details.png)

*Note.* LR denotes learning rate; WD denotes weight decay; Eval batch denotes the evaluation batch size. The final three-path operation uses `limit_x` and `scale_x`; all three paths are evaluated for every sample.

Default values are defined in `train.py`.

- `batch_size`: 8.
- `eval_batch_size`: 16.
- `epochs`: 50.
- `optimizer`: AdamW.
- `lr`: `5e-5`.
- `weight_decay`: `1e-4`.
- `hidden_dim`: 160.
- `heads`: 4.
- `feature_layers`: 2.
- `align_layers`: 2.
- `dropout`: 0.15.
- `ema_decay`: 0.997, starting from epoch 4.
- `scale_i`: 0.08.
- `gain_i`: 0.20.
- `limit_x`: 0.1.
- `scale_x`: 2.0.

## Reference

```text
Package A
```
