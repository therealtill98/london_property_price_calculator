import streamlit as st
import numpy as np

# ============================================================
# MODEL COEFFICIENTS (from your regression output)
# ============================================================

INTERCEPT = 12.0915

# District coefficients (baseline is BARKING AND DAGENHAM = 0)
DISTRICT_COEFS = {
    "Barking and Dagenham": 0.0,
    "Barnet": 0.5061,
    "Bexley": -0.0021,
    "Brent": 0.4873,
    "Bromley": 0.2158,
    "Camden": 0.9720,
    "City of Westminster": 1.1314,
    "Croydon": 0.1701,
    "Ealing": 0.6078,
    "Enfield": 0.2751,
    "Greenwich": 0.2348,
    "Hackney": 0.5452,
    "Hammersmith and Fulham": 0.8916,
    "Haringey": 0.4806,
    "Harrow": 0.5170,
    "Havering": 0.2189,
    "Hillingdon": 0.4808,
    "Hounslow": 0.8297,
    "Islington": 0.8117,
    "Kensington and Chelsea": 1.3506,
    "Kingston upon Thames": 0.4367,
    "Lambeth": 0.5339,
    "Lewisham": 0.2862,
    "Merton": 0.6090,
    "Newham": 0.1453,
    "Redbridge": 0.3897,
    "Richmond upon Thames": 0.8861,
    "Southwark": 0.5504,
    "Sutton": 0.7107,
    "Tower Hamlets": 0.5969,
    "Waltham Forest": 0.2545,
    "Wandsworth": 0.6959,
}

# Property type coefficients (baseline is "Detached" = 0)
PROPERTY_TYPE_COEFS = {
    "Detached House": 0.1595,
    "Detached Bungalow": 0.1721,
    "Semi-Detached House": 0.0766,
    "Semi-Detached Bungalow": 0.1135,
    "Terraced House": 0.0437,
    "Flat": -0.1121,
    "Maisonette": -0.1612,
    "House (unspecified)": -0.1583,
}

# Other coefficients
COEF_OLD_NEW = 0.2695  # 1 if new build
COEF_TOTAL_FLOOR_AREA_SCALED = 0.2029
COEF_NUMBER_HABITABLE_ROOMS_SCALED = 0.0439

# You'll need to replace these with actual values from your training data
# These are placeholders - use the mean and std from your StandardScaler
FLOOR_AREA_MEAN = 86.99212822933303
FLOOR_AREA_STD = 53.563372321806376
ROOMS_MEAN = 3.936350357824273
ROOMS_STD = 1.7875930814226857


def predict_price(district: str, property_type: str, floor_area: float, 
                  num_rooms: int, is_new_build: bool) -> float:
    """
    Predict house price using the linear regression coefficients.
    Returns price in GBP.
    """
    # Start with intercept
    log_price = INTERCEPT
    
    # Add district effect
    log_price += DISTRICT_COEFS.get(district, 0.0)
    
    # Add property type effect
    log_price += PROPERTY_TYPE_COEFS.get(property_type, 0.0)
    
    # Add new build effect
    if is_new_build:
        log_price += COEF_OLD_NEW
    
    # Add scaled continuous variables
    floor_area_scaled = (floor_area - FLOOR_AREA_MEAN) / FLOOR_AREA_STD
    rooms_scaled = (num_rooms - ROOMS_MEAN) / ROOMS_STD
    
    log_price += COEF_TOTAL_FLOOR_AREA_SCALED * floor_area_scaled
    log_price += COEF_NUMBER_HABITABLE_ROOMS_SCALED * rooms_scaled
    
    # Convert from log price to actual price
    price = np.exp(log_price)
    
    return price


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="London House Price Estimator",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 London House Price Estimator")
st.markdown("Get an instant price estimate for properties across London boroughs.")

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    district = st.selectbox(
        "Borough",
        options=sorted(DISTRICT_COEFS.keys()),
        index=sorted(DISTRICT_COEFS.keys()).index("Hackney")
    )
    
    property_type = st.selectbox(
        "Property Type",
        options=list(PROPERTY_TYPE_COEFS.keys()),
        index=2  # Semi-Detached House
    )
    
    is_new_build = st.toggle("New Build", value=False)

with col2:
    floor_area = st.number_input(
        "Total Floor Area (m²)",
        min_value=15,
        max_value=500,
        value=75,
        step=5
    )
    
    num_rooms = st.slider(
        "Number of Habitable Rooms",
        min_value=1,
        max_value=10,
        value=4
    )

st.divider()

# Calculate and display prediction
if st.button("Get Price Estimate", type="primary", use_container_width=True):
    price = predict_price(district, property_type, floor_area, num_rooms, is_new_build)
    
    st.success(f"### Estimated Price: £{price:,.0f}")
    
    # Show breakdown
    with st.expander("See how this was calculated"):
        st.markdown(f"""
        **Model breakdown (log-scale):**
        - Base price: {INTERCEPT:.4f}
        - Borough effect ({district}): {DISTRICT_COEFS.get(district, 0):.4f}
        - Property type effect ({property_type}): {PROPERTY_TYPE_COEFS.get(property_type, 0):.4f}
        - New build effect: {COEF_OLD_NEW if is_new_build else 0:.4f}
        - Floor area effect: {COEF_TOTAL_FLOOR_AREA_SCALED * (floor_area - FLOOR_AREA_MEAN) / FLOOR_AREA_STD:.4f}
        - Rooms effect: {COEF_NUMBER_HABITABLE_ROOMS_SCALED * (num_rooms - ROOMS_MEAN) / ROOMS_STD:.4f}
        
        The model predicts log(price), which is then converted to GBP.
        """)

st.divider()

st.caption(
    "⚠️ This is a demo model for educational purposes. "
    "Estimates are based on historical transaction data and EPC records. "
    "Always consult professional valuers for actual property decisions."
)
