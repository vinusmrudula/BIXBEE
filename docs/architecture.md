# TerraLens System Architecture

TerraLens follows a modular pipeline for satellite image super-resolution using SwinIR.

## Architecture Overview

```text
Sentinel-2 Satellite Image
          |
          v
   Image Preprocessing
          |
          v
   SwinIR SR Model
          |
          v
  4x Super-Resolution
          |
          +------------------+
          |                  |
          v                  v
 Image Evaluation     Uncertainty Analysis
          |                  |
          +--------+---------+
                   |
                   v
          TerraLens Interface
                   |
                   v
        Enhanced Satellite Image
