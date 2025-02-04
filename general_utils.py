from sklearn.preprocessing import MinMaxScaler
from matplotlib.dates import DateFormatter
from pickle import dump, load
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def normalize_data(df, range, target_column):

    data = df.copy()
    target_df_series = df[[target_column]]
    
    X_scaler = MinMaxScaler(feature_range=range)
    y_scaler = MinMaxScaler(feature_range=range)
    
    X_scale_dataset = X_scaler.fit_transform(data)
    y_scale_dataset = y_scaler.fit_transform(target_df_series)
    
    dump(X_scaler, open('X_scaler.pkl', 'wb'))
    dump(y_scaler, open('y_scaler.pkl', 'wb'))

    return (X_scale_dataset,y_scale_dataset)


def batch_data(x_data, y_data, batch_size, predict_period):
    X_batched, y_batched, yc = [], [], []

    for i in range(0,len(x_data),1):
        x_value = x_data[i: i + batch_size]
        y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
        yc_value = y_data[i: i + batch_size]
        if len(x_value) == batch_size and len(y_value) == predict_period:
            X_batched.append(x_value)
            y_batched.append(y_value)
            yc.append(yc_value)

    return np.array(X_batched), np.array(y_batched), np.array(yc)

def split_train_test(data):
    train_size = len(data) - 20
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

def predict_index(dataset, X_train, batch_size, prediction_period):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[batch_size: X_train.shape[0] + batch_size + prediction_period, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + batch_size:, :].index

    return train_predict_index, test_predict_index

def plot_stock_price(final_df, stock_name):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(final_df['Date'], final_df['Close'], color='#0000FF')
    ax.set(xlabel="Date", ylabel="USD", title=f"{stock_name} Stock Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.show()

def plot_results(Real_test_price, Predicted_test_price, stock_name):
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))

    Predicted_test_price = Predicted_test_price.reshape(-1, 1)
    
    rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

    idx = np.random.randint(rescaled_Predicted_price.shape[0], size=20)
    
    predicted = rescaled_Predicted_price[idx]
    real = rescaled_Real_price[idx]

    RMSE = np.sqrt(mean_squared_error(predicted, real))
    mse = mean_squared_error(predicted, real)
    mae = mean_absolute_error(predicted, real)
    print('Test RMSE: ', RMSE)
    print('Test MSE:', mse)
    print('Test MAE:', mae)
    
    plt.figure(figsize=(10, 5))
    plt.plot(real, color='#0000FF')
    plt.plot(predicted, color = '#FF0000', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left")
    plt.title(f"Prediction on test data for {stock_name}")
    plt.show()

def plot_test_data(Real_test_price, Predicted_test_price, index_test, output_dim, stock_name):
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    test_predict_index = index_test

    rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=test_predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
  
    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=test_predict_index[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)
  
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('Test RMSE: ', RMSE)
    
    plt.figure(figsize=(10, 5))
    plt.plot(real_price["real_mean"], color='#0000FF')
    plt.plot(predict_result["predicted_mean"], color = '#FF0000', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left")
    plt.title(f"Prediction on test data for {stock_name}")
    plt.show()