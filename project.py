import streamlit as st
import pandas as pd

# ΒΗΜΑ 1ο: Φόρτωση Δεδομένων: Η εφαρμογή θα πρέπει να είναι σε θέση να φορτώνει tabular data (csv)
def load_data():
    uploaded_file = st.file_uploader("heart_failure_clinical_records.csv")
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.success("File loaded successfully!")    
        return data

# ΒΗΜΑ 2ο: Προδιαγραφές Πίνακα
def validate_data(data):
    if data is not None:
        rows, cols = data.shape
        if cols < 2:
            st.error("The data must have at least one feature column and one label column")
            return False
        
        # Assuming the last column is the label
        features = data.iloc[:, :-1]
        label = data.iloc[:, -1]
        
        if label.nunique() < 2:
            st.error("The label column must contain at least two unique values")
            return False
        
        st.success(f"Data has {rows} samples and {cols-1} features with 1 label column.")
        return True
    return False

def main():
    st.title("Application Development for Data Analysis")

    # Load the data
    data = load_data()

    # Display the data
    if data is not None:
        if validate_data(data): 
            st.write("Data Preview:")
            st.dataframe(data.head())
    else:
            st.error("Invalid data structure. Please upload a dataset with the correct format.")

if __name__ == "__main__":
    main()
