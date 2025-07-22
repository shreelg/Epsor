# Epsor

---

## Video Demo

[Epsor Demo](https://drive.google.com/file/d/14Juhbu8TcNUrKTbFWoC51FJjV3eEr5ZP/view?usp=sharing)

---

## Overview

**Epsor** is an EPS prediction tool that leverages a combination of **Conditional GANs (CGAN)** and **XGBoost** to forecast stock movements around earnings announcements, utilizing a comprehensive set of financial features.

- **CGAN:** Generates synthetic EPS data to augment limited historical datasets, effectively capturing diverse market dynamics not readily available from public data sources.  
- **XGBoost:** Trains on this enhanced dataset with carefully engineered EPS-related features, resulting in improved predictive accuracy (F1 score: 84.65%).  
- **SHAP:** Provides interpretability by highlighting the key features influencing each model prediction.  
- **FastAPI:** Implements a lightweight API that accepts ticker inputs and returns buy/sell signals alongside confidence scores and detailed explanations.
- **SQLite:** for large data creation and management
- All features used: Ticker, Sector, Industry, Shares Outstanding, Beta, PE Ratio, Earnings Date, Market Cap, Revenue, price and volume changes, volatility metrics, moving averages (ma_5, ma_10, ma_20), EPS actual/estimate/surprise values, and more. 


## Project Structure

![File Structure](https://github.com/user-attachments/assets/f8482d77-645b-4928-b830-24e3e923f6cb)
