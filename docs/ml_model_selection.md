# ML Model Selection for EBSD Phase Classification

Date: 2026-03-02

## 1. Selection Goal

Select practical, open-source, high-performing classification backbones that are easy to implement for single-channel 16-bit Kikuchi patterns while preserving reproducibility.

This shortlist targets strong transfer learning support first, then scalability.

## 2. Selection Criteria

- Open-source implementation and active ecosystem support.
- Pretrained weights available for transfer learning.
- Feasible training/inference cost on lab GPU/CPU setups.
- Robustness potential on texture-rich grayscale patterns.
- Easy integration in this repo via `timm`.

## 3. Recommended Shortlist (5 Models)

### 1) ConvNeXt V2 (recommended default family)

- Paper: [ConvNeXt V2](https://arxiv.org/abs/2301.00808)
- Official code: [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
- Practical pretrained model ID: `convnextv2_nano.fcmae_ft_in22k_in1k`
- Example weights: [timm/convnextv2_nano.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k)
- Why: strong modern convolutional inductive bias with stable transfer behavior.

### 2) EfficientNetV2

- Paper: [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- Practical pretrained model ID: `tf_efficientnetv2_s.in21k_ft_in1k`
- Example weights: [timm/tf_efficientnetv2_s.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k)
- Why: strong accuracy/efficiency balance and mature deployment footprint.

### 3) MobileNetV4 (efficient modern baseline)

- Paper: [MobileNetV4](https://arxiv.org/abs/2404.10518)
- Practical pretrained model ID: `mobilenetv4_conv_medium.e500_r224_in1k`
- Example weights: [timm/mobilenetv4_conv_medium.e500_r224_in1k](https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r224_in1k)
- Why: efficient and recent architecture, useful when runtime/memory constraints matter.

### 4) CoAtNet (conv-attention hybrid)

- Paper: [CoAtNet](https://arxiv.org/abs/2106.04803)
- Practical pretrained model ID: `coatnet_0_rw_224.sw_in1k`
- Example weights: [timm/coatnet_0_rw_224.sw_in1k](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k)
- Why: combines convolutional locality with attention-based global modeling.

### 5) MaxViT (multi-axis + local/global attention)

- Paper: [MaxViT](https://arxiv.org/abs/2204.01697)
- Practical pretrained model ID: `maxvit_rmlp_tiny_rw_256.sw_in1k`
- Example weights: [timm/maxvit_rmlp_tiny_rw_256.sw_in1k](https://huggingface.co/timm/maxvit_rmlp_tiny_rw_256.sw_in1k)
- Why: high-capacity option for complex texture discrimination.

## 4. Optional Sixth Candidate

- Swin V2 paper: [Swin Transformer V2](https://arxiv.org/abs/2111.09883)
- Example model: `swinv2_tiny_window8_256.ms_in1k`
- Example weights: [timm/swinv2_tiny_window8_256.ms_in1k](https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k)

## 5. Chosen Initial Training Order

1. `convnextv2_nano.fcmae_ft_in22k_in1k` (pretrained, then scratch baseline).
2. `tf_efficientnetv2_s.in21k_ft_in1k` (pretrained, then scratch baseline).
3. `mobilenetv4_conv_medium.e500_r224_in1k` (pretrained).
4. `coatnet_0_rw_224.sw_in1k` (pretrained).
5. `maxvit_rmlp_tiny_rw_256.sw_in1k` (pretrained).

Reason for this order:

- Starts with strongest practical CNN-first transfer candidates.
- Adds efficiency-focused and hybrid architectures for robustness comparison.
- Keeps all runs in one implementation stack (`timm`) for fairer comparison.

## 6. Pretrained and Scratch Policy

For each shortlisted model:

- Run A: pretrained initialization (`pretrained=true`).
- Run B: scratch initialization (`pretrained=false`).

This directly measures transfer benefit on EBSD data and avoids over-claiming pretrained gains.

## 7. Reproducibility Note

Model availability for these IDs was verified in this environment via `timm.list_models(pretrained=True)` at implementation time. Primary model implementations are provided by [rwightman/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models).
