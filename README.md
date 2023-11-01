Here is a draft README that provides an overview of your code for modeling mouse visual cortex using Transformers:

# Transformers for Modeling Mouse Visual Cortex 

This repository contains code to implement a Transformer neural network model for predicting neural activity in mouse primary visual cortex (V1) based on visual stimuli videos. The model is trained and evaluated on two-photon calcium imaging data from the Allen Brain Observatory.

## Model overview

The model takes in a sequence of video frame snippets and mouse running speed as input. It encodes the video frames using a convolutional patch embedding layer followed by a spatial Transformer encoder to capture visual features. 

The running speed modulates the representations via an embedding layer. A temporal Transformer encoder then models the temporal dynamics. Finally, a global average pooling layer aggregates features across time, which are passed to a dense output layer to predict V1 neural activity.

Key components:

- Input sequence of video snippets and running speed
- Conv2D patch embedding 
- Spatial Transformer encoder
- Running speed modulation
- Temporal Transformer encoder
- Global average pooling 
- Output dense layer for prediction

## Data

The data comes from the Allen Brain Observatory, a collection of in-vivo two-photon calcium imaging recordings from mouse visual cortex under visual stimulation. 

- The videos are natural movie snippets used as stimuli during the recordings.

- The targets are df/f traces from imaged neurons, preprocessed and normalized.

- Running speed is also utilized as a contextual signal.

- Data is loaded using the AllenSDK.

## Training

- Adam optimizer with learning rate scheduling 
- Mean absolute error loss
- Batch training with generator to feed sequenced data

## Evaluation

- Model generates predictions for held-out validation video snippets
- Quantitative comparison of predicted and true neural traces

## Usage

The main scripts are:

- `model.py` - Defines the core Transformer model architecture
- `data.py` - Loads and preprocesses the Allen Institute dataset  
- `train.py` - Main training loop
- `eval.py` - Functions for evaluating predictions

To run an example training:

```
python train.py
```

## References

Relevant papers on neural encoding models:

- Walker et al. Nature 2021
- Cadena et al. PLoS Comp Bio 2019
- Batty et al. Nature Neuro 2016

Let me know if you would like any sections expanded or additional details covered! Also happy to incorporate any other code files or results you want highlighted.
