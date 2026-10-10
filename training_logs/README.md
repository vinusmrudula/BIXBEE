# TerraLens Training Documentation

This directory documents the model training and experimentation performed during development of the TerraLens super-resolution pipeline.

## Model

The primary super-resolution architecture used by TerraLens is SwinIR (Swin Transformer for Image Restoration).

The final trained model used by the prototype is stored in the repository as:

`ntro26142_swinir_x4_final.pth`

## Super-Resolution Scale

The model is configured for:

**4x Super-Resolution**

## Development Workflow

The model development workflow consists of:

1. Satellite data preparation
2. Input preprocessing
3. SwinIR model configuration
4. Model training/fine-tuning
5. Model checkpoint generation
6. Validation and visual inspection
7. Selection of final model weights
8. Integration with the TerraLens inference pipeline

## Experiment Notebook

The model experimentation and development workflow is available in:

`NTRO_26142_SwinIR_fixed.ipynb`

The notebook provides the reproducible experimentation environment used during development.

## Final Model

The final model checkpoint used for inference is:

`ntro26142_swinir_x4_final.pth`

This checkpoint is loaded by the TerraLens inference pipeline to perform satellite image super-resolution.

## Reproducibility

Dependencies required to run the implementation are provided in the root-level:

`requirements.txt`

Future experiments and training summaries can be added to this directory without modifying the main application.
