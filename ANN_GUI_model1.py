import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# Load the trained ANN model
model = tf.keras.models.load_model('your_model.h5')

# App title
st.title("RC Shear Wall Energy Dissipation Prediction")
st.write("Predict the NCDE for reinforced concrete shear walls using trained ANN.")

# Input field labels (with descriptions)
input_labels = [
    "Wall Length (lw, mm)", 
    "Wall Height (hw, mm)", 
    "Wall Thickness (tw, mm)", 
    "Concrete Compressive Strength (f′c, MPa)", 
    "Yield Strength of Tension Reinforcement (fyt, MPa)", 
    "Yield Strength of Shear Reinforcement (fysh, MPa)", 
    "Yield Strength of Longitudinal Reinforcement (fyl, MPa)", 
    "Yield Strength of Boundary Layer Reinforcement (fybl, MPa)", 
    "Reinforcement Ratio for Tension (ρt)", 
    "Shear Reinforcement Ratio (ρsh)", 
    "Longitudinal Reinforcement Ratio (ρl)", 
    "Boundary Layer Reinforcement Ratio (ρbl)", 
    "Axial Load Ratio (P/(Agf′c))", 
    "Boundary Layer Width (b0, mm)", 
    "Reinforcement Diameter (db, mm)", 
    "Spacing-to-Diameter Ratio (s/db)", 
    "Aspect Ratio (AR)", 
    "Moment-to-Shear Ratio (M/Vlw)"
]

# Collect inputs from the user
st.sidebar.header("Enter Input Parameters")
inputs = []
for label in input_labels:
    value = st.sidebar.number_input(label, value=0.0, step=0.1)
    inputs.append(value)

# Reshape inputs for prediction
inputs = np.array(inputs).reshape(1, -1)

# Prediction button
if st.sidebar.button("Predict NCDE"):
    try:
        # Make prediction
        prediction = model.predict(inputs)
        st.success(f"Predicted NCDE: {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Save inputs button
if st.sidebar.button("Save Inputs"):
    try:
        file_name = st.text_input("Enter filename to save inputs (e.g., inputs.csv):")
        if file_name:
            pd.DataFrame([inputs.flatten()], columns=input_labels).to_csv(file_name, index=False)
            st.success(f"Inputs saved successfully to {file_name}.")
    except Exception as e:
        st.error(f"Error in saving inputs: {str(e)}")

# Load inputs button
if st.sidebar.button("Load Inputs"):
    try:
        uploaded_file = st.file_uploader("Upload a CSV file with input parameters", type=["csv"])
        if uploaded_file is not None:
            loaded_data = pd.read_csv(uploaded_file)
            st.write("Loaded Inputs:")
            st.write(loaded_data)
            inputs = loaded_data.iloc[0].values.reshape(1, -1)
    except Exception as e:
        st.error(f"Error in loading inputs: {str(e)}")

# Log section
st.header("Logs")
log_messages = []

# Functionality to log inputs and predictions
def log_prediction(inputs, prediction):
    global log_messages
    log_entry = f"Inputs: {inputs.flatten()} -> Predicted NCDE: {prediction[0][0]:.2f}"
    log_messages.append(log_entry)

# Display logs
for log in log_messages:
    st.text(log)

# Help/Info section
st.sidebar.header("Help/Info")
st.sidebar.info("""
### Input Field Descriptions:
- **lw**: Wall length (mm)
- **hw**: Wall height (mm)
- **tw**: Wall thickness (mm)
- **f'c**: Concrete compressive strength (MPa)
- **fyt**: Yield strength of tension reinforcement (MPa)
- **fysh**: Yield strength of shear reinforcement (MPa)
- **fyl**: Yield strength of longitudinal reinforcement (MPa)
- **fybl**: Yield strength of boundary layer reinforcement (MPa)
- **ρt**: Reinforcement ratio for tension
- **ρsh**: Shear reinforcement ratio
- **ρl**: Longitudinal reinforcement ratio
- **ρbl**: Boundary layer reinforcement ratio
- **P/(Agf′c)**: Axial load ratio
- **b0**: Boundary layer width (mm)
- **db**: Diameter of reinforcement (mm)
- **s/db**: Spacing-to-diameter ratio
- **AR**: Aspect ratio
- **M/Vlw**: Moment-to-shear ratio
""")
