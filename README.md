# shopper-purchase-intent-ml

Instructions for building this repository step by step.

## Setup

1. Use Python 3.12.
2. Create and activate a virtual environment.

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies once `requirements.txt` has been added.

   ```bash
   pip install -r requirements.txt
   ```

## Build order

1. Add `requirements.txt` and the dataset under `data/`.
2. Add the notebook and `test_data.csv`.
3. Add the trained models under `model/` and the evaluation output under `results/`.
4. Add `app.py` (the Streamlit app).
5. Replace this file with the final README.

## Run the app

```bash
streamlit run app.py
```
