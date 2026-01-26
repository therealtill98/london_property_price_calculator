import streamlit as st
import numpy as np

# Model coefficients from training output

INTERCEPT = 12.0915

# District coefficients 
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

FLOOR_AREA_MEAN = 86.99212822933303
FLOOR_AREA_STD = 53.563372321806376
ROOMS_MEAN = 3.936350357824273
ROOMS_STD = 1.7875930814226857

# Model error (for confidence interval)
RMSE_LOG = 0.4313


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

# STREAMLIT UI

st.set_page_config(
    page_title="London Property Price Estimator",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 London Property Price Estimator")
st.markdown("Get an instant price estimate for properties across London boroughs.")
st.markdown("*Price estimates are generated through a proprietary ML model trained on +1M historic sale prices that were matched to official energy certificate records.*")

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
    
    is_new_build = st.radio(
        "Property Age",
        options=["Existing Property", "New Build"],
        index=0,
        horizontal=True
    ) == "New Build"

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
    
    # Calculate ±1 std dev range
    log_price = np.log(price)
    price_low = np.exp(log_price - RMSE_LOG)
    price_high = np.exp(log_price + RMSE_LOG)
    
    st.success(f"### Estimated Price: £{price:,.0f}")
    st.markdown(f"*Likely range: £{price_low:,.0f} – £{price_high:,.0f}*")
    
    # Show breakdown
    with st.expander("See how this was calculated"):
        base_price = np.exp(INTERCEPT)
        district_multiplier = np.exp(DISTRICT_COEFS.get(district, 0))
        property_multiplier = np.exp(PROPERTY_TYPE_COEFS.get(property_type, 0))
        new_build_multiplier = np.exp(COEF_OLD_NEW) if is_new_build else 1.0
        floor_area_scaled = (floor_area - FLOOR_AREA_MEAN) / FLOOR_AREA_STD
        rooms_scaled = (num_rooms - ROOMS_MEAN) / ROOMS_STD
        floor_area_multiplier = np.exp(COEF_TOTAL_FLOOR_AREA_SCALED * floor_area_scaled)
        rooms_multiplier = np.exp(COEF_NUMBER_HABITABLE_ROOMS_SCALED * rooms_scaled)
        
        st.markdown(f"""
        **Price breakdown:**
        - Base price: £{base_price:,.0f}
        - Borough adjustment ({district}): ×{district_multiplier:.2f}
        - Property type adjustment ({property_type}): ×{property_multiplier:.2f}
        - New build adjustment: ×{new_build_multiplier:.2f}
        - Floor area adjustment ({floor_area} m²): ×{floor_area_multiplier:.2f}
        - Rooms adjustment ({num_rooms} rooms): ×{rooms_multiplier:.2f}
        
        **Final estimate:** £{base_price:,.0f} × {district_multiplier:.2f} × {property_multiplier:.2f} × {new_build_multiplier:.2f} × {floor_area_multiplier:.2f} × {rooms_multiplier:.2f} = **£{price:,.0f}**
        """)

st.divider()

st.caption(
    "⚠️ This is a demo for educational purposes. "
    "Always consult professional valuers for actual property decisions."
)
