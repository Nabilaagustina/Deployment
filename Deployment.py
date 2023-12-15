import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# mendeklarasi dataset
df_proces = pd.read_csv('https://raw.githubusercontent.com/Nabilaagustina/Deployment/main/dataset/jabodetabek_house_price.csv')

# menghapus kolom url 
df_proces = df_proces.drop('url', axis=1)

# menghapus kolom title
df_proces = df_proces.drop('title', axis=1)

# menghapus kolom address karena sudah memiliki makna sama dengan kolom city
df_proces = df_proces.drop('address', axis=1)

# menghapus kolom district karena sudah memiliki makna sama dengan kolom city
df_proces = df_proces.drop('district', axis=1)

# menghapus kolom lan karena sudah memiliki makna sama dengan kolom city
df_proces = df_proces.drop('lat', axis=1)

# menghapus kolom long karena sudah memiliki makna sama dengan kolom city
df_proces = df_proces.drop('long', axis=1)

# menghapus kolom property_type karena semua data hanya memiliki 1 varian saja
df_proces = df_proces.drop('property_type', axis=1)

# menghapus ads_id title
df_proces = df_proces.drop('ads_id', axis=1)

# menghapus kolom building_age karena terdapat banyak data NaN
df_proces = df_proces.drop('building_age', axis=1)

# menghapus kolom year_built karena terdapat banyak data NaN
df_proces = df_proces.drop('year_built', axis=1)

# menghapus kolom building_orientation karena terdapat banyak data NaN
df_proces = df_proces.drop('building_orientation', axis=1)

# menghapus kolom facilities karena isinya sudah terwakili dengan kolom furnishing
df_proces = df_proces.drop('facilities', axis=1)

# Menghapus data NaN dan data outliner
df_proces = df_proces.dropna()

cols = list(df_proces.describe().T.index)

# List untuk menyimpan hasil
results = []

for col in cols:
    q1 = df_proces[col].quantile(0.25)
    q3 = df_proces[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_after = df_proces[(df_proces[col] < lower_bound) | (df_proces[col] > upper_bound)]
    percent_outliers_after = (len(outliers_after)/len(df_proces)) * 100
    results.append({'Kolom': col, 'Persentase Outliers': percent_outliers_after})
    df_proces[col] = df_proces[col][(df_proces[col] >= lower_bound) & (df_proces[col] <= upper_bound)]

df_proces = df_proces.dropna()

# Manipulasi data
# Encoder
# menghapus spasi yang tidak berguna pada nilai setiap baris pada kolom city
df_proces['city'] = df_proces['city'].str.strip()

# encoder kolom city
encoder_city = LabelEncoder()

# mengapilkasikan ke dalam kolom city
encoder_city.fit(df_proces['city'])

# mengubah data kolom city
df_proces['city'] = encoder_city.transform(df_proces['city'])

# menghapus spasi yang tidak berguna pada nilai setiap baris pada kolom certificate
df_proces['certificate'] = df_proces['certificate'].str.strip()

# encoder kolom certificate
encoder_certificate = LabelEncoder()

# mengapilkasikan ke dalam kolom certificate
encoder_certificate.fit(df_proces['certificate'])

# mengubah data kolom certificate
df_proces['certificate'] = encoder_certificate.transform(df_proces['certificate'])

# Menghapus keterangan 'mah' pada nilai kolom 'electricity' dan hanya mengambil angka
df_proces['electricity'] = df_proces['electricity'].str.replace('mah', '').str.extract('(\d+)')

# Menghapus nilai kosong ('') pada kolom 'electricity'
df_proces = df_proces.dropna()

# menghapus spasi yang tidak berguna pada kolom electricity
df_proces['electricity'] = df_proces['electricity'].str.strip()

# menghapus spasi yang tidak berguna pada nilai setiap baris pada kolom property_condition
df_proces['property_condition'] = df_proces['property_condition'].str.strip()

# encoder kolom property_condition
encoder_property_condition = LabelEncoder()

# mengapilkasikan ke dalam kolom property_condition
encoder_property_condition.fit(df_proces['property_condition'])

# mengubah data kolom property_condition
df_proces['property_condition'] = encoder_property_condition.transform(df_proces['property_condition'])

# menghapus spasi yang tidak berguna pada nilai setiap baris pada kolom furnishing
df_proces['furnishing'] = df_proces['furnishing'].str.strip()

# encoder kolom furnishing
encoder_furnishing = LabelEncoder()

# mengapilkasikan ke dalam kolom furnishing
encoder_furnishing.fit(df_proces['furnishing'])

# mengubah data kolom furnishing
df_proces['furnishing'] = encoder_furnishing.transform(df_proces['furnishing'])

df_proces = df_proces.drop_duplicates()

feature = df_proces.iloc[:, 1:]
target = df_proces[['price_in_rp']]

mutual_info = mutual_info_regression(feature, target, random_state=42)

mutual_info = pd.Series(mutual_info)
mutual_info.index = feature.columns
mutual_info = mutual_info.sort_values(ascending=False)

mutual_info = pd.DataFrame(mutual_info)
mutual_info = mutual_info.reset_index()
mutual_info = mutual_info.rename(columns={'index':'label', 0:'score'})

best_feature = mutual_info.nlargest(10, 'score')
lbl_best_feature = list(best_feature.label)
lbl_best_feature.insert(0, 'price_in_rp')

df_proces = df_proces[lbl_best_feature]
X = df_proces.drop('price_in_rp', axis=1).copy()
y = df_proces[['price_in_rp']].copy()

sc_10_feature = StandardScaler()
sc_10_feature.fit(X)

df_proces.iloc[:, 1:] = sc_10_feature.transform(X)

X = df_proces.iloc[:, 1:]
y = df_proces.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

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
            df['city'] = encoder_city.transform(df['city'])
            df['property_condition'] = encoder_property_condition.transform(df['property_condition'])
            df = sc_10_feature.transform(df)
            prediction = model.predict(df)[0]
            st.success(f'House price predictions is:&emsp;{prediction:,}')