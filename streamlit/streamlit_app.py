import streamlit as st
import pandas as pd
import naampy
from naampy import in_rolls_fn_gender, predict_fn_gender
import base64


# Define your sidebar options
sidebar_options = {
    'Append Electoral Roll Data to First Name': in_rolls_fn_gender,
    'Predict Using the Model': predict_fn_gender
}
       
def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("naampy: Infer Sociodemographic Characteristics from Indian Names")

    # Generic info.
    st.write('Using data from the Indian Electoral Rolls, we estimate the proportion female, male, and third sex for a particular first name, year, and state.')
    st.write('[Github](https://github.com/appeler/naampy)')

    # Set up the sidebar
    st.sidebar.title('Select Function')
    selected_function = st.sidebar.selectbox('', list(sidebar_options.keys()))

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")

        if selected_function == "Append Electoral Roll Data to First Name": 
            fname_col = st.selectbox("Select column with first name", df.columns)
            state = st.selectbox("Select a state", ["andaman", "andhra", "arunachal", "assam",
                                                "bihar", "chandigarh", "dadra", "daman", "delhi",
                                                "goa", "gujarat", "haryana", "himachal", "jharkhand",
                                                "jk", "karnataka", "kerala", "maharashtra", "manipur",
                                                "meghalaya", "mizoram", "mp", "nagaland", "odisha",
                                                "puducherry", "punjab", "rajasthan", "sikkim", "tripura",
                                                "up", "uttarakhand"])
            function = sidebar_options[selected_function]
            if st.button('Run'):
                transformed_df = function(df, namecol=fname_col, state = state)
                st.dataframe(transformed_df)
                download_file(transformed_df)
    
        elif selected_function == "Predict Using the Model":
            fname_col = st.selectbox("Select column with first name", df.columns)
            function = sidebar_options[selected_function]
            if st.button('Run'):
                transformed_df = function(df[fname_col])
                st.dataframe(transformed_df)
                download_file(transformed_df)
        
# Run the app
if __name__ == "__main__":
    app()
