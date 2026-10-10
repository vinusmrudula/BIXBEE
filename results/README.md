# TerraLens Experimental Results

This directory contains experimental outputs produced by the TerraLens super-resolution pipeline.

## Evaluation Strategy

TerraLens evaluates super-resolved imagery using both visual and quantitative analysis.

The objective is not only to generate visually sharper imagery, but also to examine whether the reconstructed output remains consistent with the underlying satellite observation.

## Visual Evaluation

Outputs are inspected by comparing:

1. Original medium-resolution input
2. Super-resolved SwinIR output
3. Reference imagery where available

Representative results are stored in this directory and in the repository's sample outputs.

## Quantitative Evaluation

The evaluation pipeline is designed to support image-quality measurements such as:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

Where appropriate reference data is available, these metrics can be used to quantify reconstruction quality.

## Baseline Comparison

A conventional interpolation-based upscaling method can be used as a baseline when evaluating the deep-learning reconstruction.

This allows the SwinIR result to be compared against simple image enlargement rather than evaluating it in isolation.

## Uncertainty Analysis

TerraLens also investigates prediction uncertainty through repeated stochastic inference.

Multiple model predictions can be generated for the same input and their variation can be analyzed.

Low variation indicates relatively stable reconstruction, while higher variation identifies regions where the model output should be interpreted with greater caution.

## Result Samples

Representative input and super-resolved samples are provided in the repository.

Additional quantitative results will be documented here as experiments are performed.
