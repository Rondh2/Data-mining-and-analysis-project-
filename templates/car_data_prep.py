import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
from datetime import datetime
import pickle

def prepare_data(df, encoder_path='encoder.pkl', scaler_path='scaler.pkl', fit=False, feature_names_path='feature_names.pkl'):
    # Data arrangement
    df['capacity_Engine'] = df['capacity_Engine'].replace('None', np.nan)
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '')
    df['Km'] = df['Km'].replace('None', np.nan)
    df['Km'] = df['Km'].str.replace(',', '')
    df['capacity_Engine'] = df['capacity_Engine'].fillna(0)
    df = df.drop(columns=['Test', 'Supply_score', 'Cre_date', 'Repub_date', 'Area', 'Description'], errors='ignore')

    def clean_model_column(row):
        manufactor = row['manufactor']
        model = row['model']
        model = re.sub(r'\b' + re.escape(manufactor) + r'\b', '', model, flags=re.IGNORECASE)
        model = re.sub(r'\(\d{4}\)', '', model)
        model = re.sub(r'\b\d{4}\b', '', model)
        model = re.sub(r',', '', model)
        model = model.strip()

        return model

    df.loc[:, 'model'] = df.apply(clean_model_column, axis=1)

    # arranging types
    df['Year'] = df['Year'].astype(int)
    df['Hand'] = df['Hand'].astype(int)
    df['Gear'] = df['Gear'].astype('category')
    df['capacity_Engine'] = df['capacity_Engine'].astype(int)
    df['Engine_type'] = df['Engine_type'].astype('category')
    df['Prev_ownership'] = df['Prev_ownership'].astype('category')
    df['Curr_ownership'] = df['Curr_ownership'].astype('category')
    df['City'] = df['City'].astype('string')

    # Filling in missing values
    df['Engine_type'].fillna('בנזין', inplace=True)
    df['Gear'].fillna('אוטומטית', inplace=True)
    df['Prev_ownership'].fillna('פרטית', inplace=True)
    df['Curr_ownership'].fillna('פרטית', inplace=True)

    # Calculation of Km according to the annual average of a vehicle in Israel
    current_year = datetime.now().year
    df['difference'] = current_year - df['Year']
    df['Km'].fillna(df['difference'] * 15000, inplace=True)
    df.drop(columns=['difference'], inplace=True)

    # One-Hot Encoding
    columns_to_encode = ['Color', 'manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'City']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if fit:
        encoded_features = encoder.fit_transform(df[columns_to_encode])
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
    else:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        encoded_features = encoder.transform(df[columns_to_encode])

    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns_to_encode))
    df = df.drop(columns_to_encode, axis=1).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    # Add missing features with value 0
    feature_names = encoder.get_feature_names_out(columns_to_encode)
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    # scaling
    scaler = StandardScaler()
    columns_to_scale = ['Year', 'Hand', 'capacity_Engine', 'Km']
    if fit:
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    df.fillna(0, inplace=True)

    if fit:
        feature_order = list(df.columns)
        with open(feature_names_path, 'wb') as f:
            pickle.dump(feature_order, f)
    else:
        with open(feature_names_path, 'rb') as f:
            feature_order = pickle.load(f)
        feature_order = [f for f in feature_order if f in df.columns]  # Ensure all features exist in the DataFrame
        df = df.reindex(columns=feature_order, fill_value=0)

    return df
