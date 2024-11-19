import streamlit as st
import pandas as pd
import xgboost as xgb
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

X = data.drop(['price_scaled', axis = 1])
y = data['price_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.form(
  property_type	= st.pills('Property Type', ['Flat', 'House', 'Penthouse', 'Upper Portion', 'Farm House',
         'Lower Portion', 'Room'])
  #location = 
  city = st.pills('City', ['Islamabad', 'Karachi', 'Faisalabad', 'Lahore', 'Rawalpindi'])
  baths = st.sliders('Baths', X.baths.min(), X.baths.max())
  purpose = st.pills('Purpose', ['For Sale', 'For Rent'])
  bedrooms = st.sliders('Bedrooms', X.bedrooms.min(), X.bedrooms.max())
  Area Type = st.pills('Area Type', ['Marla', 'Kanal'])
  Area Size = st.number_input('Area Size', 0, 1000)
st.form_submit_button(label="Submit")

def input():
  property_type	= st.segmented_control('Property Type', ['Flat', 'House', 'Penthouse', 'Upper Portion', 'Farm House',
       'Lower Portion', 'Room']
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = input()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
# Apply Model to Make Prediction
prediction = model.predict(df)

# Dự đoán (trong không gian đã scale)
y_pred_scaled = model.predict(X_scaled)

# Đảo ngược scale cho kết quả dự đoán
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))



st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
