# Import library python
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# Import model 
model_rf = joblib.load('./model_final.bin')

# Import encoder
# ['price_in_rp', 'building_size_m2',
# 'land_size_m2', 'electricity',
# 'bathrooms', 'bedrooms',
# 'city', 'maid_bathrooms',
# 'maid_bedrooms', 'floors',
# 'property_condition'] 

# Yang membutuhkan encoder adalah kolom city dan property_condition
enc_city = joblib.load('./encoder_city.joblib')
enc_property_condition = joblib.load('./encoder_property_condition.joblib')

# Import scaller
sc_10_feature = joblib.load('./sc_10_feature.joblib')

# Membuat header 
st.markdown("<h1 style='text-align:center'>House Price Prediction</h1>", unsafe_allow_html=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            col3, col4 = st.columns(2)
            with col3:
                building_size_m2 = st.number_input(label='Insert the building size (m2)', min_value=0)
            with col4:
                land_size_m2 = st.number_input(label='Insert the land size (m2)', min_value=0)
        with st.container():
            col5, col6 = st.columns(2)
            with col5:
                electricity = st.number_input(label='Insert capacity electric', min_value=0)
            with col6:
                bathrooms = st.number_input(label='Insert total of bathrooms', min_value=0)
        with st.container():
            col7, col8= st.columns(2)
            with col7:
                bedrooms = st.number_input(label='Insert total of bedrooms', min_value=0)
            with col8:
                city = st.selectbox("Insert the city location", ('Bekasi', 'Bogor', 'Depok', 'Jakarta Barat', 'Jakarta Selatan', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat', 'Tangerang'), label_visibility=st.session_state.visibility, disabled=st.session_state.disabled)
        with st.container():
            col9, col10 = st.columns(2)
            with col9:
                maid_bathrooms = st.number_input(label='Insert total of maid\'s bathrooms', min_value=0)
            with col10:
                maid_bedrooms = st.number_input(label='Insert total of maid\'s bedrooms', min_value=0)
        with st.container():
            col11, col12= st.columns(2)
            with col11:
                floors = st.number_input(label='Insert total of floors', min_value=0)
            with col12:
                property_condition = st.selectbox("House condition", ('bagus', 'bagus sekali', 'baru', 'sudah renovasi', 'butuh renovasi'), label_visibility=st.session_state.visibility, disabled=st.session_state.disabled)

    with col2:
        image_house = Image.open(r"./Images/House.jpg")
        st.image(image_house, caption='House', width=325)

def user_input():
    data = {
        'building_size_m2': building_size_m2,
        'land_size_m2': land_size_m2,
        'electricity': electricity,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'city': city,
        'maid_bathrooms': maid_bathrooms,
        'maid_bedrooms': maid_bedrooms,
        'floors': floors,
        'property_condition': property_condition,
    }
    input_data = pd.DataFrame(data, index=[0])
    return input_data

df = user_input()
if st.button('Buat Prediksi'):  
    with st.container():
        col13, col14 = st.columns(2)
        with col13:
            st.dataframe(df.astype(str).T.rename(columns={0:'input_data'}))
        with col14:
            df['city'] = enc_city.transform(df['city'])
            df['property_condition'] = enc_property_condition.transform(df['property_condition'])
            df = sc_10_feature.transform(df)
            st.write(df)
            prediction = model_rf.predict(df)[0]
            st.success(f'House price predictions is:&emsp;{prediction:,}')