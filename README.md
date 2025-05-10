# Time Series Forecasting (Quality Engineering)

A modular and reproducible pipeline for univariate time series forecasting using ARIMA, LSTM, and Transformer models. This repository is designed for plug-and-play usage on electricity demand data, but can be easily adapted to other time series datasets.


This repo includes the following autoregressive forecasting models:

- **ARIMA**: Classical linear model with seasonal extensions.
- **LSTM**: Recurrent neural network capable of learning temporal dependencies.
- **Transformer**: A causal, decoder-style transformer model with masked self-attention and positional encoding, tailored for univariate forecasting.

Experiments are run on UK national electricity demand data (NESO, 2024), with all models predicting one step ahead over a 7-day test window (336 half-hourly steps).

<img width="600" alt="image" src="https://github.com/user-attachments/assets/717a9a6e-1cfd-43ba-9642-aff7a5a3baf3" />

