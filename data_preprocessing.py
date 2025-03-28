import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df['AQI_7d_avg'] = df['AQI'].rolling(window=7).mean().bfill()
    df['AQI_30d_avg'] = df['AQI'].rolling(window=30).mean().bfill()

    city_encoder = LabelEncoder()
    df['City_encoded'] = city_encoder.fit_transform(df['City'])

    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                 'O3', 'Benzene', 'Toluene', 'Xylene', 'Hour_sin',
                 'Hour_cos', 'City_encoded', 'AQI_7d_avg', 'AQI_30d_avg']

    target = 'AQI'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    train_data = df.iloc[:int(0.7 * len(df))][features]
    scaler_features.fit(train_data)
    scaler_target.fit(df.iloc[:int(0.7 * len(df))][[target]])

    df[features] = scaler_features.transform(df[features])
    df[target] = scaler_target.transform(df[[target]])

    return df, features, target
