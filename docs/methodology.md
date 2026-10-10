# TerraLens Methodology

TerraLens is a deep-learning-based super-resolution pipeline developed for SIH26142 to enhance medium-resolution satellite imagery while preserving meaningful spatial information.

## 1. Input Satellite Imagery

The pipeline uses medium-resolution Sentinel-2 satellite imagery as the primary input for super-resolution processing.

## 2. Preprocessing

Input imagery is prepared for inference by converting it into the format and dimensions required by the super-resolution model. The preprocessing stage ensures consistent input before enhancement.

## 3. SwinIR Super-Resolution

TerraLens uses SwinIR (Swin Transformer for Image Restoration) as its primary super-resolution architecture.

The model uses transformer-based image restoration to learn both local and long-range spatial relationships within satellite imagery.

A 4× super-resolution configuration is used to generate a higher-resolution representation of the input image.

## 4. Super-Resolution Inference

The preprocessed image is passed through the trained SwinIR model.

The inference pipeline:

1. Loads the trained model weights.
2. Prepares the input image.
3. Performs 4× super-resolution.
4. Reconstructs the enhanced image.
5. Generates the final output for visualization and analysis.

## 5. Evaluation

The generated super-resolution imagery is evaluated using quantitative image-quality metrics and visual comparison with reference imagery where available.

The evaluation stage is used to determine whether enhancement improves image reconstruction while maintaining consistency with the original satellite observation.

## 6. Uncertainty Estimation

TerraLens incorporates uncertainty estimation using multiple stochastic inference passes.

Instead of relying only on a single model output, repeated predictions can be used to estimate the variability of the reconstructed image.

This provides an indication of regions where the model prediction is relatively stable and regions where the generated reconstruction should be interpreted with greater caution.

## 7. Visualization

The TerraLens interface allows users to inspect the original and super-resolved satellite imagery for direct visual comparison.

The complete pipeline can therefore be summarized as:

**Sentinel-2 Input → Preprocessing → SwinIR 4× Super-Resolution → Evaluation → Uncertainty Analysis → TerraLens Visualization**
