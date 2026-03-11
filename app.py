import streamlit as st
from src.inference import predict_property

st.title("MolecuGraph AI")
st.write("Predict HOMO-LUMO Gap for your molecule.")

smiles = st.text_input("Enter SMILES string:")
if st.button("Predict"):
    try:
        prediction = predict_property(smiles)
        st.success(f"Predicted HOMO-LUMO Gap: {prediction:.4f} eV")
    except ValueError as e:
        st.error(str(e))