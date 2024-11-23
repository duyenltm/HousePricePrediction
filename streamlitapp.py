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
uni = data
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

st.write(data)

X = data.drop(['price_scaled'], axis=1)
y = data['price_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def input():
  with st.form("input_form"):
    st.header("Enter House Details")
    property_type = st.pills('Property Type', uni['property_type'].unique())
    location = st.selectbox('Location', uni['location'].unique())
    city = st.pills('City', uni['city'].unique())
    baths = st.slider('Baths',1,7)
    purpose = st.pills('Purpose', uni['purpose'].unique())
    bedrooms = st.slider('Bedrooms',1,7)
    area_size = st.number_input('Area Size',1,1000)
    area_type = st.pills('Area Type', uni['area_type'].unique())
    area = area_size * 272.51 if area_type == 'Marla' else area_size * 5445
    submitted = st.form_submit_button("Submit")
    if submitted:
      data = {'property_type': property_type,
              'location': location,
              'city': city,
              'purpose': purpose,
              'area_scaled': area,
              'baths_scaled': baths,
              'bedrooms_scaled': bedrooms}
      return pd.DataFrame(input_data, index=[0])
    else:
      return None

df = input()
st.write(df)
df_scaled = scaler.fit_transform(df)

Tree_reg = RandomForestRegressor(random_state=42)
Tree_reg.fit(X_train, y_train)
y_pred = Tree_reg.predict(X_test)
joblib.dump(Tree_reg, 'best_model.pkl')
best_model = joblib.load('best_model.pkl')

prediction = best_model.predict(df_scaled)

price = y.inverse_transform(prediction.reshape(-1, 1))

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
