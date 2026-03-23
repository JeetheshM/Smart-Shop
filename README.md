# Smart-Shop

Clean and attractive Streamlit dashboard for ecommerce conversion prediction.

## What this project does

- Loads ecommerce session data from `shop_smart_ecommerce` (or `online_shoppers.csv` if present).
- Trains a decision tree pipeline with preprocessing (scaling + one-hot encoding).
- Shows dashboard KPIs and model quality metrics.
- Accepts user inputs in a structured form and predicts purchase likelihood.

## Files

- `app.py`: Streamlit frontend and model workflow.
- `shop_smart_ecommerce`: dataset used by the app.
- `requirements.txt`: dependencies.
- `shop_smaart.ipynb`: original notebook file.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

After running, open:

- http://localhost:8501

## Dashboard sections

- Overview KPIs (rows, features, conversion rate, F1).
- Session Input form (numeric + categorical controls).
- Model Dashboard (accuracy, F1, classification report, confusion matrix).
- Prediction Output with conversion probability.