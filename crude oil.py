import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
print("HORMUZ PULSE: ML PIPELINE FOR TABLEAU")
## READING 4 FILES
df_crude = pd.read_csv("/users/mohammedthoufeeque/downloads/crude oil - sheet1.csv")
df_inflation = pd.read_csv("/users/mohammedthoufeeque/downloads/india energy_inflation.csv")
df_currency = pd.read_csv("/users/mohammedthoufeeque/downloads/india energy_currency.csv")
df_oil_price = pd.read_csv("/users/mohammedthoufeeque/downloads/india energy_oil price.csv")
print(df_crude.head())
## MAPPING YEARLY CONFIGURATION
df_crude['Date'] = pd.to_datetime(df_crude['Date'])
df_crude['Year'] = df_crude['Date'].dt.year
print(df_crude.head())
print("Inflation Data")
print(df_inflation.head())
print("\n Currency Data")
print(df_currency.head())
print("\n Oil Price Data")
## MERGING
print(df_oil_price.head())
df_merged = pd.merge(df_crude,df_inflation,on='Year',how='inner')
df_merged = pd.merge(df_merged,df_currency,on='Year',how='inner')
df_merged = pd.merge(df_merged,df_oil_price,on='Year',how='inner')
print("\n Final merged data")
print(df_merged.head())
## FEATURE ENGINEERING
print("\n Engineering features for predictive logic")
df_merged['Crude_Price_Lag_1M'] = df_merged['Crude_Oil_Price'].shift(1)
df_merged.dropna(inplace=True)
## MODEL TRAINING(HISTORICAL LEARNING)
print("\n Training Random Forest Regressor on historical trends")
## set target
x = df_merged[['Crude_Oil_Price','Crude_Price_Lag_1M','USD_INR','Crude_Imports']]
y = df_merged[['Petrol','Diesel']] 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x, y)
print("> Success! Baseline model training completed")
print(df_merged.columns)

# # 2026 CRISIS SCENARIO SIMULATION
print("\n Simulating 2026 Strait of Hormuz chokepoint crisis")

## tablueue scenarios
scenarios = pd.DataFrame({
    'Scenario_ID': [1, 2, 3],
    'Scenario_Name': [
        '1. Base Case (Normal Trade)', 
        '2. Partial Blockage (Medium Crisis)', 
        '3. Full Blockage (Severe 2026 Crisis)'
    ],
    'Crude_Oil_Price': [75.0, 105.0, 135.0],      # crude price increase
    'Crude_Price_Lag_1M': [73.0, 95.0, 120.0],
    'USD_INR': [83.5, 85.5, 88.0],                # value of price decrease
    'Crude_Imports': [10635815, 9040442, 6381489]  # supply(0%, 15%, 40%)
})

# prediction
predicted_fuel = model.predict(scenarios[['Crude_Oil_Price', 'Crude_Price_Lag_1M', 'USD_INR', 'Crude_Imports']])
scenarios['Predicted_Petrol'] = predicted_fuel[:, 0]
scenarios['Predicted_Diesel'] = predicted_fuel[:, 1]

# # BUSINESS IMPACT STRUCTURING

print("\n Computing business impact metrics for Tableau")

# difference of base price & crisis price
base_p = scenarios.loc[0, 'Predicted_Petrol']
severe_p = scenarios.loc[2, 'Predicted_Petrol']

fuel_hike_pct = ((severe_p - base_p) / base_p) * 100
logistics_impact = fuel_hike_pct * 0.40 

## business matrix
scenarios['Fuel_Hike_Percentage'] = fuel_hike_pct
scenarios['Logistics_Impact_Percentage'] = logistics_impact

## EXPORT FOR TABLEAU VISUALIZATION
print("\n Exporting final dataset to CSV for Tableau")

# file imported into tableue
scenarios.to_csv("hormuz_2026_simulated_output.csv", index=False)

print("[FINISHED] Run successful! File 'hormuz_2026_simulated_output.csv generated")
##  install prophet
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet import Prophet
print("prepping data for prophet")
forecast_df = df_merged[['Date','Crude_Oil_Price']].rename(columns={'Date':'ds','Crude_Oil_Price': 'y'})
## train and make prophet model
model_prophet = Prophet()
model_prophet.fit(forecast_df)
## future frame in next 30 days
future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)
## print future forecast
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
##save futureforecasted data
forecast.to_csv('hormuz_2026_prophet_forecast.csv',index=False)
print("[FINISHED] prophet forecast saved successfully")
