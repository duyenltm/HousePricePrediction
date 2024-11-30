import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

st.title('Pakistan House Price Prediction')
st.write('---')

st.markdown("### 🏠 House Price Prediction 🏡")

url = 'https://drive.google.com/file/d/1HPzLNrEIBduaatuEsf7EDWJ2_J0f4T2N/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
data = pd.read_csv(url)

data = data.drop(columns=['property_id','page_url', 'location_id', 'province_name','area', 'latitude', 
                          'longitude', 'date_added', 'agency', 'agent','Area Category' ])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.drop(data[data['price']==0].index, inplace = True)
data.drop(data[data['baths']==0].index, inplace = True)
data.drop(data[data['bedrooms']==0].index, inplace = True)
data.drop(data[data['Area Size']==0].index, inplace = True)
uni = data
data['area'] = data.apply(lambda row: row['Area Size'] * 25.2929 if row['Area Type'] == 'Marla' 
                          else row['Area Size'] * 505.858, axis=1)
data = data.drop(columns=['Area Type', 'Area Size'])

X_data = data.select_dtypes(include=['float64', 'int64'])
Q1 = X_data.quantile(0.25)
Q3 = X_data.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (X_data < (Q1 - 1.5 * IQR)) | (X_data > (Q3 + 1.5 * IQR))
data = data[~outlier_condition.any(axis=1)]
data.reset_index(drop=True, inplace=True)

scaler_price = MinMaxScaler()
data['price'] = scaler_price.fit_transform(data[['price']])

scaler_features = MinMaxScaler()
data[['area', 'baths', 'bedrooms']] = scaler_features.fit_transform(data[['area', 'baths', 'bedrooms']])

label_encoders = {'purpose': LabelEncoder(),
                  'property_type': LabelEncoder(),
                  'city': LabelEncoder(),
                  'location': LabelEncoder()}

for col in label_encoders.keys():
    data[col] = label_encoders[col].fit_transform(data[col])

X = data.drop('price', axis = 1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def input():
  with st.form("input_form"):
    st.header("Enter House Details")
    property_type = st.pills('Select Property Type', uni['property_type'].unique())
    location = st.selectbox('Select Location', uni['location'].unique())
    city = st.pills('Select City', uni['city'].unique())
    baths = st.slider('Select Baths',1,7)
    purpose = st.pills('Select Purpose', uni['purpose'].unique())
    bedrooms = st.slider('Select Bedrooms',1,7)
    area = st.number_input('Enter Area Size (m²)',1.00 ,600.00 , step=0.01)
    submitted = st.form_submit_button("Submit")
    if submitted:
      data = {'property_type': property_type,
              'location': location,
              'city': city,
              'baths': baths,
              'purpose': purpose,
              'bedrooms': bedrooms,
              'area': area}
      return pd.DataFrame([data])
    else:
      return None

df = input()
st.write(df)

Ranfor_reg = RandomForestRegressor(random_state=42)
Ranfor_reg.fit(X_train, y_train)

def process_input(df):
  for col, le in label_encoders.items():
    df[col] = le.transform(df[col])
  df[['area', 'baths', 'bedrooms']] = scaler_features.transform(df[['area', 'baths', 'bedrooms']])
  return df

if df is not None:
  df_processed = process_input(df)
  df_processed = df_processed.values.reshape(1, -1)
  prediction = Ranfor_reg.predict(df_processed)
  price = prediction[0] * (scaler_price.data_max_[0] - scaler_price.data_min_[0]) + scaler_price.data_min_[0]
  st.success("Prediction complete!")
  st.subheader(f"The predicted house price is: {price:.2f}")
