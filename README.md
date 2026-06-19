# 🏨 Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![Model Accuracy](https://img.shields.io/badge/Accuracy-80.1%25-brightgreen)](https://github.com/YonathanHH/Predicting_Hotel_Cancellations_in_Portugal) [![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.876-brightgreen)](https://github.com/YonathanHH/Predicting_Hotel_Cancellations_in_Portugal) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://haryhotelprediction.streamlit.app/)

## Overview

This project develops a machine learning model to predict whether customers are likely to cancel their hotel bookings at a Portuguese hotel. It implements an end-to-end ML workflow — from exploratory data analysis and preprocessing to model development, evaluation, and deployment through an interactive Streamlit application with SHAP-based explainability.

**Target Problem:** Binary classification to identify booking cancellations before they occur, enabling the hotel to implement proactive retention strategies and optimise resource allocation.

---

## Business Context

Hotel cancellations directly erode revenue and complicate operational planning. A false negative (missing a real canceller) means the hotel loses a booking it could have saved with early intervention. A false positive (flagging a non-canceller) wastes retention resources but has a lower cost. This asymmetry justifies prioritising **Recall** and **ROC-AUC** over raw accuracy in model selection.

**Stakeholders:** Hotel management, revenue management team, customer service department  
**Impact:** Proactive outreach to high-risk bookings can reduce revenue losses and improve occupancy forecasting.

---

## Dataset

- **Source:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. [https://doi.org/10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126)
- **Records:** 119,390 bookings from two Portuguese hotels (2015–2017)
- **Raw data:** Place `Hotel_booking_demand.csv` inside the `data/` directory (see [Dataset Setup](#dataset-setup))

`Target`: `is_canceled` — Whether the booking was cancelled (1) or not (0)

`Selected Features` (chosen based on business interpretability and feature importance):
| Feature | Description |
|---|---|
| Market Segment | Distribution channel group |
| Customer Type | Type of booking (Transient, Contract, etc.) |
| Tourist Origin | Local vs. International guest |
| Booking Changes | Number of amendments made |
| Previous Cancellations | Guest's cancellation history |
| Parking Spaces Requirement | Whether parking was requested |
| Special Requests | Number of special requests made |
| Waiting List | Days the booking spent on a waiting list |

---

## Key Findings

- **Online TA bookings** carry the highest cancellation risk — the indirect booking channel reduces guest commitment
- **Guests with prior cancellation history** are the strongest single predictor of future cancellations
- **Special requests correlate with lower cancellation rates** — guests who invest effort in their stay are more likely to follow through
- **Waiting list entries** dramatically increase cancellation probability once the wait exceeds 30 days

---

## Project Structure

```
Predicting_Hotel_Cancellations_in_Portugal/
│
├── README.md                                    # Project documentation
├── Hotel_Cancelation_End_to_End_ML.ipynb        # End-to-end ML notebook
├── final_model.sav                              # Trained XGBoost pipeline (pickle)
├── app.py                                       # Streamlit web application
├── requirements.txt                             # Pinned dependencies
├── .gitignore                                   # Excludes data files and binaries
└── data/                                        # Local data directory (not tracked by Git)
    ├── Hotel_booking_demand.csv                 # Raw dataset (download separately)
    └── hotel_dataset_cleaned.csv                # Cleaned/transformed dataset
```

---

## Dataset Setup

The CSV files are excluded from Git due to their size (~4MB each). To run the notebook locally:

1. Download `Hotel_booking_demand.csv` from the [original publication](https://doi.org/10.1016/j.dib.2018.11.126) or [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
2. Place it in the `data/` directory: `data/Hotel_booking_demand.csv`
3. Run the notebook — the cleaned CSV will be generated at `data/hotel_dataset_cleaned.csv`

---

## Final Model

| Hyperparameter | Value |
|---|---|
| Algorithm | XGBoost |
| Learning Rate | 0.27 |
| Max Depth | 12 |
| n_estimators | 185 |

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.801 |
| Precision | 0.750 |
| Recall | 0.693 |
| F1-Score | 0.720 |
| ROC-AUC | **0.876** |

---

## Running Locally

### Prerequisites
- Python 3.8+ (developed on Python 3.13.9)
- pip or conda

### Setup
```bash
git clone https://github.com/YonathanHH/Predicting_Hotel_Cancellations_in_Portugal.git
cd Predicting_Hotel_Cancellations_in_Portugal
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

### Quick Inference (Python)
```python
import pandas as pd
import pickle

df = pd.read_csv("data/hotel_dataset_cleaned.csv")
with open("final_model.sav", "rb") as f:
    pipe = pickle.load(f)

print('Predicted class :', pipe.predict(df[51:55]))
print('Predicted proba :', pipe.predict_proba(df[51:55]))
```

---

## Conclusions & Recommendations

The XGBoost model achieves a ROC-AUC of 0.876, meaning it correctly ranks a cancelling booking above a non-cancelling one 87.6% of the time. Practically, the hotel can:

1. **Flag high-risk bookings** (probability > 0.6) for proactive outreach at time of booking
2. **Prioritise retention offers** to Online TA guests with prior cancellations
3. **Reduce waiting list duration** as a lever — long waits dramatically increase cancellation rates
4. **Reward special requests** — guests who make special requests are more engaged; early acknowledgment reinforces commitment

---

## Project Metadata

- **Project Type:** ML Capstone — Machine Learning (Module 3)
- **Problem Type:** Binary Classification
- **Model:** XGBoost
- **Created:** December 2025
- **Author:** Yonathan Hary Hutagalung
- **Institution:** Purwadhika Digital Technology School

---

## References

- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- Scikit-learn: https://scikit-learn.org/
- Streamlit: https://docs.streamlit.io/
- SHAP: https://shap.readthedocs.io/
