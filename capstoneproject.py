# Library
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Configure the page
st.set_page_config(
    page_title="Electricity Price Forecasting",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data df
with open('model_electricity_price.sav', 'rb') as file:
    model = pickle.load(file)

# Sidebar and Selectbox
add_selectbox = st.sidebar.selectbox('Select a page', ('Home', 'Dataset', 'Exploratory Data Analysis', 'Model Building', 'Prediction'))

# Load dataset
df_energy = pd.read_csv('D:\dibimbing.id\CapstoneProject\energy_price\energy_dataset.csv')

df_weather = pd.read_csv('D:\dibimbing.id\CapstoneProject\energy_price\weather_features.csv')


# Home
if add_selectbox == 'Home':
    st.title('Electricity Price Prediction')
    st.markdown('This is a simple application to predict electricity price using deep learning.')
    st.markdown('Dataset used can be accesed [here](https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda)')
    st.image('https://assets.amigoenergy.com/wp-content/uploads/2020/12/electricity-how-it-works-all-about-electrons-over-city-image.jpg', width=500)
    st.write('''
    **Project Overview**
''')
    

# Dataset
elif add_selectbox == 'Dataset':
    st.title('Dataset Information')
    st.write('''
    Data consist of various sources (fossils and renewable energy) of power generation, weather in five different cities in Spain,
    and total electricity price from the used fuels.
    ''')

    # Show dataset
    st.write('''
    **Show Dataset**
    ''')
    # Load energy dataset
    #df_energy = pd.read_csv('D:\dibimbing.id\CapstoneProject\energy_price\energy_dataset.csv')
    # Convert time to datetime object and set it as index
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True, infer_datetime_format=True)
    df_energy = df_energy.set_index('time')

    # Load weather dataset
    #df_weather = pd.read_csv('D:\dibimbing.id\CapstoneProject\energy_price\weather_features.csv')
    # Convert dt_iso to datetime type, rename it and set it as index
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True, infer_datetime_format=True)
    df_weather = df_weather.drop(['dt_iso'], axis=1)
    df_weather = df_weather.set_index('time')

    # Split the df_weather into 5 dataframes (one for each city)
    df_valencia, df_madrid, df_bilbao, df_barcelona, df_seville = [x for _, x in df_weather.groupby('city_name')]
    dfs = [df_valencia, df_madrid, df_bilbao, df_barcelona, df_seville]

    # Merge all dataframes into the final dataframe
    df_final = df_energy
    for df in dfs:
        city = df['city_name'].unique()
        city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
        df = df.add_suffix('_{}'.format(city_str))
        df_final = df_final.merge(df, on=['time'], how='outer')
        df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)
    df_final

    # Show dataset shape
    st.write(f'''
    **Dataset Shape:** {df_final.shape}
    ''')

    # Show dataset description
    st.write('''
    **Dataset Description**
    ''')
    st.dataframe(df_final.describe())

    # Show price actual line graph 
    rolling = df_final['price actual'].rolling(24*7*4, center=True).mean()
    # Display actual price and monthly rolling mean line chart with labels
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_final['price actual'], label='Hourly')
    ax.plot(rolling, label='Monthly Rolling Mean')
    ax.set_ylabel('Actual Price (€/MWh)')
    ax.set_title('Actual Hourly Electricity Price and Monthly Rolling Mean')
    ax.legend()
    st.pyplot(fig)

#EDA
elif add_selectbox == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')
    st.write('''
    **Outlier**
    ''')
    global numeric_cols 
    views = st.selectbox('Select handling', ('', 'Check outlier', 'After handling outlier', ''))
    numeric_cols = list(df_weather.select_dtypes(include=['int64', 'float64']).columns)

    if views == 'Check outlier':
        x_values = st.selectbox('X axis', options=numeric_cols)
        plot = px.box(data_frame=df_weather, x=x_values)
        plot.show()

    elif views == 'After handling outlier':
            # Create list column will be transformed
            cols_to_transform = ['temp', 'temp_min', 'temp_max', 'pressure',
                                 'wind_speed', 'rain_1h', 'rain_3h', 'snow_3h']

            # Lakukan iterasi atas log transform pada setiap kolom
            for col in cols_to_transform:
                 df_weather[col] = np.log1p(df_weather[col])
            # Fill null values using interpolation
            df_weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
    
            x_after = st.selectbox('X axis', options=cols_to_transform)
            plot = px.box(data_frame=df_weather, x=x_after)
            plot.show()
    
#Model Building
elif add_selectbox == 'Model Building':
    st.header('Model Building')
    st.write('''
    Projecting electricity price, LSTM method will be employed in this case. 
    ''')

# Convert time to datetime object and set it as index
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True, infer_datetime_format=True)
    df_energy = df_energy.set_index('time')

    # Convert dt_iso to datetime type, rename it and set it as index
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True, infer_datetime_format=True)
    df_weather = df_weather.drop(['dt_iso'], axis=1)
    df_weather = df_weather.set_index('time')

    # Split the df_weather into 5 dataframes (one for each city)
    df_valencia, df_madrid, df_bilbao, df_barcelona, df_seville = [x for _, x in df_weather.groupby('city_name')]
    dfs = [df_valencia, df_madrid, df_bilbao, df_barcelona, df_seville]

    # Merge all dataframes into the final dataframe
    df_final = df_energy
    #df_final = df_final[df_final.drop(columns=['weather_main', 'weather_description']).values]
    for df in dfs:
        city = df['city_name'].unique()
        city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
        df = df.add_suffix('_{}'.format(city_str))
        df_final = df_final.merge(df, on=['time'], how='outer')
        df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)
    df_final

    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
            
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
            
            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i : i + target_size])

        return np.array(data), np.array(labels)
    
    train_end_idx = 27048
    cv_end_idx = 31056
    test_end_idx = 35064

    #st.write(df_final.columns)

    X = df_final[df_final.drop(columns=['price actual', 'weather_']).values]
    y = df_final['price actual'].values
    y = y.reshape(-1, 1)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(X[:train_end_idx])
    scaler_y.fit(y[:train_end_idx])

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)

    pca = PCA(n_components=0.80)
    pca.fit(X_norm[:train_end_idx])
    X_pca = pca.transform(X_norm)

    dataset_norm = np.concatenate((X_pca, y_norm), axis=1)
    past_history = 24
    future_target = 0

    X_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                        0, train_end_idx, past_history, 
                                        future_target, step=1, single_step=True)

    X_test, y_test = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                    cv_end_idx, test_end_idx, past_history, 
                                    future_target, step=1, single_step=True)

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train = train.cache().shuffle(1000).batch(32).prefetch(1)

    y_test = y_test.reshape(-1, 1)
    y_test_inv = scaler_y.inverse_transform(y_test)

    def plot_model_rmse_and_loss(history):
        
        train_rmse = history.history['root_mean_squared_error']
        val_rmse = history.history['val_root_mean_squared_error']
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.plot(train_rmse, label='Training RMSE')
        plt.plot(val_rmse, label='Validation RMSE')
        plt.legend()
        plt.title('Epochs vs. Training and Validation RMSE')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Epochs vs. Training and Validation Loss')
        plt.show()

    # Define some common parameters

    input_shape = X_train.shape[-2:]
    loss = tf.keras.losses.MeanSquaredError()
    metric = [tf.keras.metrics.RootMeanSquaredError()]
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 10**(epoch / 10))
    early_stopping = tf.keras.callbacks.EarlyStopping()


    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
        LSTM(256, input_shape=input_shape, 
            return_sequences=True),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    'model.h5', monitor=('val_loss'), save_best_only=True)
    optimizer = tf.keras.optimizers.Adam(lr=6e-3, amsgrad=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    # Display model summary
    st.header('Model Summary')
    st.text(model.summary())



#Prediction
elif add_selectbox == 'Prediction':
    st.header('Prediction')
    st.write('''
    Projecting electricity price, LSTM method will be employed in this case. 
    ''')
    pred = model.predict(time)
    pred = pd.DataFrame(pred, columns=['price actual'])







    







