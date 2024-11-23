import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.title('Pakistan House Price Prediction')
st.write('---')

url = 'https://drive.google.com/file/d/1HPzLNrEIBduaatuEsf7EDWJ2_J0f4T2N/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
data = pd.read_csv(url)

data = data.drop(columns=['property_id','page_url', 'location_id', 'province_name', 'latitude', 'longitude', 'date_added', 'agency', 'agent','Area Category' ])
data.head()
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.drop(data[data['price']==0].index, inplace = True)
data.drop(data[data['baths']==0].index, inplace = True)
data.drop(data[data['bedrooms']==0].index, inplace = True)
data.drop(data[data['Area Size']==0].index, inplace = True)

data['Area Size'] = data.apply(lambda row: row['Area Size'] * 272.51
                               if row['Area Type'] == 'Marla'
                               else row['Area Size'] * 5445, axis=1)
data.drop(['Area Type'], axis = 1, inplace = True)

X_data = data.select_dtypes(include=['float64', 'int64'])
Q1 = X_data.quantile(0.25)
Q3 = X_data.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (X_data < (Q1 - 1.5 * IQR)) | (X_data > (Q3 + 1.5 * IQR))
data = data[~outlier_condition.any(axis=1)]

scaler = MinMaxScaler()
data[['price_scaled', 'area_scaled', 'baths_scaled', 'bedrooms_scaled']] = scaler.fit_transform(data[['price', 'Area Size', 'baths', 'bedrooms']])
data = data.drop(columns=['price', 'Area Size', 'baths', 'bedrooms'])

le = LabelEncoder()
data['purpose'] = le.fit_transform(data['purpose'])
data['property_type'] = le.fit_transform(data['property_type'])
data['city'] = le.fit_transform(data['city'])
data['location'] = le.fit_transform(data['location'])

X = data.drop(['price_scaled'])
y = data['price_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def input():
  property_type = st.pills('Property Type', ['Flat', 'House', 'Penthouse', 'Upper Portion', 'Farm House', 'Lower Portion', 'Room'])
        #location = st.selectbox('Location',
  city = st.segmented_control('City', ['Islamabad', 'Karachi', 'Faisalabad', 'Lahore', 'Rawalpindi'])
  baths = st.sliders('Baths', X.baths.min(), X.baths.max())
  purpose = st.segmented_control('Purpose', ['For Sale', 'For Rent'])
  bedrooms = st.sliders('Bedrooms', X.bedrooms.min(), X.bedrooms.max())
  area_type = st.segmented_control('Area Type', ['Marla', 'Kanal'])
  area_size = st.number_input('Area Size', 0, 1000)
  if area_type == 'Marla': area = area_size * 25.2929
  else: area = area_size * 505.858
  data = {'property_type': property_type,
          'location': location,
          'city': city,
          'purpose': purpose,
          'area_scaled': area,
          'baths_scaled': baths,
          'bedrooms_scaled': bedrooms}
  features = pd.DataFrame(data, index=[0])
  return features

df = input()
df_scaled = df

Ranfor_reg = RandomForestRegressor(random_state=42)
Ranfor_reg.fit(X_train, y_train)
y_pred = Ranfor_reg.predict(X_test)
joblib.dump(Tree_reg, 'best_model.pkl')
best_model = joblib.load('best_model.pkl')

prediction = best_model.predict(df_scaled)

price = scaler_y.inverse_transform(prediction.reshape(-1, 1))

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
