import os
import time
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from pcs_contract_fun import get_round_data_round_with_titles, current_epoch
import ta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import RandomizedSearchCV
from rich.console import Console
from rich.table import Table
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import KFold
import warnings
from xgboost import XGBClassifier
from skopt.space import Integer, Categorical
warnings.filterwarnings('ignore')
from tqdm import tqdm
import dask.bag as db
from dask import delayed, compute
import csv
from sklearn.model_selection import StratifiedKFold

def save_data_to_csv(data_list, file_name='historical_data2.csv'):
    with open(file_name, mode='w', newline='') as csv_file:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in data_list:
            writer.writerow(data)

    print(f"Data saved to {file_name}")


def get_history(num_previous_rounds=300):
    current_number = current_epoch() - 1
    vals = current_number - num_previous_rounds

    data_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(get_round_data_round_with_titles, i): i for i in range(current_number, vals, -1)}

        # Use tqdm to display progress and estimated time in one line
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading data", unit="round"):
            data_list.append(future.result())
    data_list.reverse()
    return data_list

def add_technical_indicators(df):
    # Fill missing values and replace infinity values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 1e8, inplace=True)
    
    df['Lock Price'] = pd.to_numeric(df['Lock Price'], errors='coerce')
    df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
    df['Trading Volume'] = pd.to_numeric(df['Trading Volume'], errors='coerce')
    
    # Adjust the window sizes for moving averages based on a 3-minute time frame
    df['SMA5'] = ta.trend.SMAIndicator(df['Close Price'], 5).sma_indicator()
    df['SMA15'] = ta.trend.SMAIndicator(df['Close Price'], 15).sma_indicator()
    
    df['EMA5'] = ta.trend.EMAIndicator(df['Close Price'], 5).ema_indicator()
    df['EMA15'] = ta.trend.EMAIndicator(df['Close Price'], 15).ema_indicator()
    
    # Add VWAP (Volume Weighted Average Price)
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['Lock Price'], df['Close Price'], df['Close Price'], df['Trading Volume']).volume_weighted_average_price()
    
    # Add other technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close Price'], 14).rsi()
    macd = ta.trend.MACD(df['Close Price'], 12, 26)
    df['MACD'] = macd.macd()
    df['MACD Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close Price'])
    df['BB High'] = bb.bollinger_hband()
    df['BB Low'] = bb.bollinger_lband()
    df['High'] = df[['Lock Price', 'Close Price']].max(axis=1)
    df['Low'] = df[['Lock Price', 'Close Price']].min(axis=1)
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close Price']).cci()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close Price'])
    df['Stoch %K'] = stoch.stoch()
    df['Stoch %D'] = stoch.stoch_signal()
    if len(df) > 1:
        if len(df) > 14:  # Check if the length of the data is greater than the window size
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Latest Price']).adx()
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Latest Price']).average_true_range()
        else:
            df['ADX'] = 0
            df['ATR'] = 0
    else:
        df['ADX'] = 0
        df['ATR'] = 0
    df['EMA10'] = ta.trend.EMAIndicator(df['Latest Price'], 10).ema_indicator()
    df['EMA30'] = ta.trend.EMAIndicator(df['Latest Price'], 30).ema_indicator()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Latest Price'], df['Trading Volume']).on_balance_volume()

    # Add Parabolic SAR
    df['Parabolic SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Latest Price']).psar()

    # Add Chaikin Money Flow (CMF)
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Latest Price'], df['Trading Volume'], 20).chaikin_money_flow()

    # Add Rate of Change (ROC)
    df['ROC'] = ta.momentum.ROCIndicator(df['Latest Price']).roc()

    # Add Williams %R
    df['Williams %R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Latest Price']).williams_r()

        # Add Ultimate Oscillator
    df['Ultimate Oscillator'] = ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Latest Price']).ultimate_oscillator()

    # Add Aroon Indicator
    aroon = ta.trend.AroonIndicator(df['Latest Price'])
    df['Aroon Up'] = aroon.aroon_up()
    df['Aroon Down'] = aroon.aroon_down()

    # Add Money Flow Index (MFI)
    df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Latest Price'], df['Trading Volume']).money_flow_index()

    # Add TRIX
    df['TRIX'] = ta.trend.TRIXIndicator(df['Latest Price']).trix()

    # Add Elder's Force Index (EFI)
    df['EFI'] = ta.volume.ForceIndexIndicator(df['Latest Price'], df['Trading Volume']).force_index()

    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def feature_selection(X, y):
    selector = SelectFromModel(estimator=XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0))
    selector.fit(X, y)
    X_selected = selector.transform(X)
    return X_selected, selector

# def hyperparameter_tuning(X, y):
#     param_distributions = {
#         'n_estimators': Integer(50, 1000),
#         'max_depth': Integer(5, 200, 'uniform'),
#         'min_child_weight': Integer(1, 50, 'uniform'),
#         'gamma': Categorical([i/10.0 for i in range(0, 5)]),
#         'subsample': Categorical([i/10.0 for i in range(6, 11)]),
#         'colsample_bytree': Categorical([i/10.0 for i in range(6, 11)]),
#         'learning_rate': Categorical([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]),
#         'objective': Categorical(['binary:logistic']),
#         'scale_pos_weight': Categorical([1, 2, 3, 4, 5, 6, 7, 8, 9])
#     }

#     bayes_search = BayesSearchCV(
#         estimator=XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0),
#         search_spaces=param_distributions,
#         n_iter=500,  # Adjust this value based on the number of iterations you want
#         cv=KFold(n_splits=5, shuffle=True, random_state=42),
#         n_jobs=-1,
#         verbose=2,
#         random_state=42,
#         scoring='accuracy'
#     )
#     bayes_search.fit(X, y)
#     best_params = bayes_search.best_params_

#     return best_params


def hyperparameter_tuning(X, y):
    param_distributions = {
        'n_estimators': Integer(50, 1000, 'log-uniform'),
        'max_depth': Integer(5, 200, 'uniform'),
        'min_child_weight': Integer(1, 50, 'uniform'),
        'gamma': Real(0, 0.5, 'uniform'),
        'subsample': Real(0.6, 1.0, 'uniform'),
        'colsample_bytree': Real(0.6, 1.0, 'uniform'),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'objective': Categorical(['binary:logistic']),
        'scale_pos_weight': Integer(1, 9, 'uniform')
    }

    bayes_search = BayesSearchCV(
        estimator=XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0),
        search_spaces=param_distributions,
        n_iter=100,  # Adjust this value based on the number of iterations you want
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring='accuracy'
    )
    bayes_search.fit(X, y)
    best_params = bayes_search.best_params_

    return best_params



def process_chunk(chunk_data_list):
    chunk_df = pd.DataFrame(chunk_data_list)
    chunk_df['Pattern'] = chunk_df['Pattern'].apply(lambda x: 1 if x == 'Bull' else 0)

    chunk_df['Lock Price'] = pd.to_numeric(chunk_df['Lock Price'])
    chunk_df['Latest Price'] = pd.to_numeric(chunk_df['Latest Price'])
    chunk_df['Lock Price - Latest Price'] = chunk_df['Lock Price'] - chunk_df['Latest Price']

    chunk_df = add_technical_indicators(chunk_df)

    return chunk_df



def trainModel():
    chunksize = 1000
    
    model, scaler, selector = None, None, None

    print("Loading data from CSV file...")
    all_chunks_df = pd.DataFrame()

    csv_file = 'historical_data.csv'

    # Create a list to store the chunk data
    chunk_data_lists = []

    # Read the CSV file in chunks
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        chunk_data_lists.append(chunk)

    print("Processing chunks...")
    # Process the chunks and concatenate the resulting dataframes
    delayed_tasks = [delayed(process_chunk)(chunk_data_list) for chunk_data_list in chunk_data_lists]
    all_chunks_df = pd.concat(compute(*delayed_tasks), ignore_index=True)

    print("Preparing data for the model...")

    X = all_chunks_df.drop(['Pattern', 'Epoch', 'Start Timestamp', 'Lock Timestamp', 'Close Timestamp'], axis=1)
    y = all_chunks_df['Pattern']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Performing feature selection...")
    X_selected, selector = feature_selection(X, y)

    print("Tuning hyperparameters...")
    best_params = hyperparameter_tuning(X_selected, y)

    print(f"Best hyperparameters: {best_params}")
    model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, tree_method='gpu_hist', gpu_id=0)

    cv_scores = cross_val_score(model, X_selected, y, cv=5)

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {cv_scores.mean():.2f}")

    model.fit(X_selected, y)

    print("Saving model and scaler...")
    model_directory = 'model_indica'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename = os.path.join(model_directory, 'random_forest_model_experiment.sav')
    scaler_filename = os.path.join(model_directory, 'scaler_experiment.sav')
    selector_filename = os.path.join(model_directory, 'selector_experiment.sav')
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    with open(scaler_filename, 'wb') as file:
        pickle.dump(scaler, file)
    with open(selector_filename, 'wb') as file:
        pickle.dump(selector, file)

def loadModel():
    model_directory = 'model_indica'
    model_filename = os.path.join(model_directory, 'random_forest_model_experiment.sav')
    scaler_filename = os.path.join(model_directory, 'scaler_experiment.sav')
    selector_filename = os.path.join(model_directory, 'selector_experiment.sav')
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename) or not os.path.exists(selector_filename):
        raise FileNotFoundError(f"Model, scaler, or selector file not found")

    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)
    with open(selector_filename, 'rb') as file:
               selector = pickle.load(file)

    return model, scaler, selector


def save_to_csv(predictions):
    output_file = 'predictions_rfi2.csv'
    columns = ['Round', 'Prediction', 'Bull Confidence', 'Bear Confidence', 'Actual']

    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_file, index=False)

    df = pd.read_csv(output_file)

    round_number, prediction, prediction_proba = predictions[0]  # Save only the first round
    bull_pct = prediction_proba[1] * 100
    bear_pct = prediction_proba[0] * 100
    new_row = pd.Series({
        'Round': round_number,
        'Prediction': prediction,
        'Bull Confidence': f"{bull_pct:.2f}%",
        'Bear Confidence': f"{bear_pct:.2f}%",
        'Actual': ''
    })

    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df.to_csv(output_file, index=False)

def predict(model, scaler, selector, rounds_ahead=2):
        while True:
            roundID = current_epoch()
            roundData = get_round_data_round_with_titles(roundID)
            roundData = [roundData]
            df_hisyxx = pd.DataFrame(roundData)
            current_time = time.time()

            # Use 'Start Timestamp' directly as it's already in UNIX timestamp format
            start_timestamp_unix = df_hisyxx.iloc[-1]['Start Timestamp']

            time_left = current_time - start_timestamp_unix

            if time_left >= 50 and time_left <= 60:
                if time_left <= 220:
                    current_round_number = current_epoch() - 1
                    print(f"Time left to start the round: {time_left} seconds")

                    historical_data = get_history(num_previous_rounds=350)
                    df_history = pd.DataFrame(historical_data)

                    predictions = []

                    for i in range(rounds_ahead):
                        df = df_history.copy()

                        df['Lock Price'] = pd.to_numeric(df['Lock Price'])
                        df['Latest Price'] = pd.to_numeric(df['Latest Price'])
                        df['Lock Price - Latest Price'] = df['Lock Price'] - \
                            df['Latest Price']

                        df = add_technical_indicators(df)

                        X = df.iloc[-1].drop(['Pattern', 'Epoch', 'Start Timestamp',
                                            'Lock Timestamp', 'Close Timestamp'])
                        X = scaler.transform([X])
                        X_selected = selector.transform(X)

                        prediction = model.predict(X_selected)[0]
                        prediction_proba = model.predict_proba(X_selected)[0]
                        prediction_str = "Bull" if prediction == 1 else "Bear"

                        predictions.append(
                            (current_round_number + i + 1, prediction_str, prediction_proba))

                        new_row = df.iloc[-1].copy()
                        new_row['Pattern'] = prediction
                        new_row['Epoch'] = current_round_number + i + 1
                        df_history = pd.concat([df_history, new_row.to_frame().T], ignore_index=True)
                    return predictions
                
                else:
                    print(f"Waiting for the next round. Time left: {time_left} seconds")
                    time.sleep(5)
            else:
                print(f"Waiting for the next round. Time left: {time_left} seconds")
                time.sleep(5)



if __name__ == '__main__':
    # # Uncomment the following line to train the model
    # trainModel()

    # Load the trained model, scaler, and selector
    model, scaler, selector = loadModel()
    console = Console()

    while True:
        # Make a prediction
        prediction_result = predict(model, scaler, selector)
        
        # Save the first round prediction to a CSV file
        save_to_csv(prediction_result)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Round", style="green", width=12)
        table.add_column("Prediction", style="green", width=12)
        table.add_column("Bull Confidence", style="green", width=20)
        table.add_column("Bear Confidence", style="green", width=20)

        for round_number, prediction, prediction_proba in prediction_result:
            bull_pct = prediction_proba[1] * 100
            bear_pct = prediction_proba[0] * 100
            table.add_row(
                str(round_number),
                prediction,
                f"{bull_pct:.2f}%",
                f"{bear_pct:.2f}%"
            )
        
        console.print(table)

    # data_list = get_history(num_previous_rounds=100000)
    # save_data_to_csv(data_list)
