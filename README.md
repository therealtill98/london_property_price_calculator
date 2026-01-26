# 🏠 London House Price Estimator

A Streamlit app that estimates London property prices based on a regression model trained on historical transaction and EPC data.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo, branch, and `app.py` as the main file
5. Click "Deploy"

Your app will be live at `https://[your-app-name].streamlit.app`

## ⚠️ Important: Update the Scaling Parameters

The `app.py` file contains placeholder values for the StandardScaler parameters:

```python
FLOOR_AREA_MEAN = 75.0  # placeholder
FLOOR_AREA_STD = 40.0   # placeholder
ROOMS_MEAN = 4.0        # placeholder
ROOMS_STD = 1.5         # placeholder
```

**You need to replace these with the actual mean and standard deviation from your training data!**

You can get these from your preprocessing pipeline:

```python
# If you used StandardScaler
print(f"FLOOR_AREA_MEAN = {scaler.mean_[floor_area_idx]}")
print(f"FLOOR_AREA_STD = {scaler.scale_[floor_area_idx]}")
```

Or calculate directly:

```python
print(f"FLOOR_AREA_MEAN = {df['TOTAL_FLOOR_AREA'].mean()}")
print(f"FLOOR_AREA_STD = {df['TOTAL_FLOOR_AREA'].std()}")
```

## Model Features Used

| Feature | Input Type | Notes |
|---------|------------|-------|
| District | Dropdown (32 boroughs) | Categorical |
| Property Type | Dropdown (8 types) | Categorical |
| Total Floor Area | Number input (m²) | Scaled |
| Number of Rooms | Slider (1-10) | Scaled |
| New Build | Toggle | Binary |

## Disclaimer

This is a demo model for educational purposes. Estimates are based on historical data and should not be used for actual property decisions.
