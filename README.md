# Emotions-Classification-via-Speech-Audio

This repository provides PyTorch implementations of four distinct models for speech emotion classification:

1. Stacked Time-Distributed 2D CNN + LSTM

2. Stacked Time-Distributed 2D CNN + Bidirectional LSTM with Attention

3. Parallel 2D CNN + Bidirectional LSTM with Attention

4. Parallel 2D CNN + Transformer Encoder

# Speech-Emotion-Classification-with-PyTorch
This repository contains PyTorch implementation of 4 different models for classification of emotions of the speech:
1. Stacked Time Distributed 2D CNN - LSTM
2. Stacked Time Distributed 2D CNN - Bidirectional LSTM with attention
3. Parallel 2D CNN - Bidirectional LSTM with attention
4. Parallel 2D CNN - Transformer Encoder

# Dataset
Models are trained on [RAVDESS Emotional Speech Audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) dataset. It consits of 1440 speech audio-only files (16 bits, 48kHz, .wav).<br />
Dataset is balanced:<br />

## Preprocessing
Signals are loaded with sample rate of 48kHz and cut off to be in the range of [0.5, 3] seconds. If the signal is shorter than 3s it is padded with zeros.<br />
**MEL spectrogram** is calculated and used as an input for the models (for the 1st and 2nd model the spectrogram is splitted into 7 chunks).<br />

Dataset is splitted into train, validation and test sets, with following percentage: (80,10,10)%.<br />
**Data augmentation** is performed by adding [Additive White Gaussian Noise](https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise) (with SNR in range [15,30]) on the original signal. This enormously improved accuracy and removed overfitting.<br />
Datasets are scaled with **Standard Scaler**.<br />
