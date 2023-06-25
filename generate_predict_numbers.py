import pandas as pd
import pyodbc
from datetime import datetime, timedelta
import statistics
from statistics import mode
from sklearn.model_selection import train_test_split
import math



connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-SJS8V3V;DATABASE=hunglotto;UID=sa;PWD=123"  
conn = pyodbc.connect(connection_string)  
cursor = conn.cursor()  

def execute_query(query, cursor, params=None):
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    result = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)

    return result if not result.empty else None


def predict(values, start_date, prediction_date):
    start_date_filter_object = datetime.strptime(start_date, '%d-%m-%Y')
    end_date_filter_object = datetime.strptime(prediction_date, '%d-%m-%Y') - timedelta(days=1)
    
    # Filter by date range
    values['date_column'] = pd.to_datetime(values['date_column'], format='%Y-%m-%d')
    values = values[(values['date_column'] >= start_date_filter_object) & (values['date_column'] <= end_date_filter_object)]
    # print(f"Data to predict in {prediction_date} from {start_date_filter_object} to {end_date_filter_object}: {values}")

    if not values.empty:
        X = values[['date_column_number']].values
        y = values['last_two_digits'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # print(f'Mean Squared Error in {prediction_date}: {mse}')
        
        input_data = (pd.to_datetime(prediction_date, format='%d-%m-%Y') - pd.Timestamp("2002-01-01")) // pd.Timedelta('1D')
        input_data = np.array([input_data]).reshape(-1, 1)
        predicted = model.predict(input_data)[0]
        
        print(f"predict {predicted} in {prediction_date}")
        return int(predicted);
    else:
        print(f"No data found at {prediction_date}")
        return None;
    
    
def get_lottery_numbers(values, date):
    # Filter by date
    date_filter = datetime.strptime(date, '%d-%m-%Y')

    filtered_values = values[values['date_column'] == date_filter].copy()

    # Filter out NULL values
    filtered_values.loc[:, 'value_column'] = filtered_values['value_column'].replace("NULL", None)
    filtered_values.loc[:, 'last_two_digits'] = filtered_values['value_column'].str[-2:]
    filtered_values.dropna(subset=['last_two_digits'], inplace=True)

    # Check if the filtered values DataFrame is not empty
    if len(filtered_values) > 0:
        return filtered_values.iloc[0]
    else:
        return None

from pyswarm import pso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def getNearestArr(number, quantity, lower_bound=0, upper_bound=99):
    if quantity % 2 == 0:
        raise ValueError("quantity must be an odd number.")
    
    half_quantity = quantity // 2
    
    start = np.maximum(lower_bound, number - half_quantity)
    end = start + quantity
    
    # Adjust start and end if end is greater than the upper bound
    if end > upper_bound + 1:
        excess = end - (upper_bound + 1)
        start -= excess
        end -= excess
        
    return np.arange(start, end)

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_statistics_with_kelly1(start_date_betting, end_date_betting, values, predict_func, get_lottery_numbers_func, values_origin, time_delta_days=1, risk_factor=1):    
    start_date_object = datetime.strptime(start_date_betting, '%d-%m-%Y')
    end_date_object = datetime.strptime(end_date_betting, '%d-%m-%Y')

    numbers_to_bet = 51
    
    result_df = pd.DataFrame(columns=['date_column', 'numbers_predict', 'winning_number'])
    
    current_date = start_date_object
    while current_date <= end_date_object:
        prediction_date = current_date.strftime('%d-%m-%Y')
        
        prediced_number = predict_func(values=values, start_date='01-01-2002', prediction_date=prediction_date)
        
        if prediced_number is not None:
            list_numbers_predicted = getNearestArr(prediced_number, numbers_to_bet)
            lottery_result = get_lottery_numbers_func(values=values_origin, date=prediction_date)
            
            if lottery_result is not None and len(list_numbers_predicted) > 0:
                result = lottery_result['last_two_digits']
                isTrue = np.isin(result, list_numbers_predicted)

                result_df = result_df._append({
                    'date_column': current_date.strftime('%d-%m-%Y'),
                    'numbers_predict': ', '.join(map(str, list_numbers_predicted)),
                    'winning_number': result
                }, ignore_index=True)

        current_date += timedelta(days=1)
        
    return result_df


def calculate_statistics_with_kelly(start_date_betting, end_date_betting, initial_bankroll, rate_of_return, values, predict_func, get_lottery_numbers_func, values_origin, time_delta_days=1, risk_factor=1):    
    start_date_object = datetime.strptime(start_date_betting, '%d-%m-%Y')
    end_date_object = datetime.strptime(end_date_betting, '%d-%m-%Y')

    numbers_to_bet = 65
    bankroll = initial_bankroll
    vay_tien = 0
    num_wins = 0
    num_losses = 0
    total_days_played = 0
    loss_streak = 0
    max_loss_streak = 0
    list_loss_streak = []
    backroll_base = int(math.floor((initial_bankroll/5)/numbers_to_bet))
    
    current_date = start_date_object
    while current_date <= end_date_object:
        prediction_date = current_date.strftime('%d-%m-%Y')
        # start_date_for_prediction = ('01-01-2002')
        #start_date_for_prediction = (current_date - timedelta(days=time_delta_days)).strftime('%d-%m-%Y')
        prediced_number = predict_func(values=values, start_date='01-01-2002', prediction_date=prediction_date)
        if prediced_number is not None:
            list_numbers_predicted = getNearestArr(prediced_number, numbers_to_bet)
            print(f"List_numbers_predicted: {prediced_number} in {list_numbers_predicted}")  
            lottery_result = get_lottery_numbers_func(values=values_origin, date=prediction_date)
            
            if lottery_result is not None and len(list_numbers_predicted) > 0:
                result = lottery_result['last_two_digits']
                isTrue = np.isin(result, list_numbers_predicted)
                print(f"Result is: {result} in {prediction_date}, STATUS {isTrue}")  
                win_probability = 1/100
                base_bet_size = backroll_base * risk_factor
                bet_size = math.floor(base_bet_size)
                bet_size = max(int(bet_size), 0)
                
                if bankroll >= bet_size*len(list_numbers_predicted):
                    if isTrue:
                        num_wins += 1
                        bankroll -= len(list_numbers_predicted)*bet_size
                        win_money = int(math.floor(bet_size * rate_of_return))
                        bankroll += win_money
                        list_loss_streak.append(loss_streak)
                        loss_streak = 0
                    else:
                        num_losses += 1

                        bankroll -= len(list_numbers_predicted)*bet_size
                        loss_streak += 1
                        if loss_streak > max_loss_streak:
                            max_loss_streak = loss_streak
                    total_days_played += 1
                    
                    print(f"loss_streak: {loss_streak} current: {bankroll} Betting {bet_size} per total bet {len(list_numbers_predicted)*bet_size} at {list_numbers_predicted} in {prediction_date}, result: {result}, status {isTrue}")
                else:
                    print("HẾT CMN TIỀN")
                    bankroll += 100000
                    vay_tien += 100000
                    
                
                
        current_date += timedelta(days=1)
        
    #DONE WHILE    
        
    net_profit_or_loss = bankroll - initial_bankroll
    
    if (total_days_played != 0):
        win_rate = num_wins / total_days_played
    else:
        win_rate = 0
        
    list_loss_streak.append(loss_streak)    
        
    # Print the results
    print(f'Total days played: {total_days_played}')
    print(f'Initial bankroll: {initial_bankroll}')
    print(f'Final bankroll: {bankroll}')
    print(f'Win rate: {win_rate * 100}%')
    print(f'Wins: {num_wins}, Losses: {num_losses}')
    print(f'Net profit/loss: {net_profit_or_loss}')
    print(f"Max lost streak {max_loss_streak}")
    print(f"List max streak: {list_loss_streak}")
    print(f"vay tien: {vay_tien}")

    return {
        'total_days_played': total_days_played,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'win_rate': win_rate,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'net_profit_or_loss': net_profit_or_loss
    }          
    
import os

def statistical_model():
    # Query data from 2002 to 2023
    start_date = '01-01-2002'
    end_date = '26-06-2023'
    start_date_object = datetime.strptime(start_date, '%d-%m-%Y')
    end_date_object = datetime.strptime(end_date, '%d-%m-%Y')

    query = """
        SELECT date_column, value_column
        FROM [hunglotto].[dbo].[lotto]
        WHERE date_column >= ? 
          AND date_column <= ?
    """
    
    values = execute_query(query, cursor, params=(start_date_object, end_date_object,))    
    
    start_date_filter = '01-01-2002'
    end_date_filter = '26-06-2023'
    start_date_filter_object = datetime.strptime(start_date_filter, '%d-%m-%Y')
    end_date_filter_object = datetime.strptime(end_date_filter, '%d-%m-%Y')
    
       # Filter out NULL values
    values['value_column'] = values['value_column'].replace("NULL", None)
    values['last_two_digits'] = values['value_column'].str[-2:]
    values = values.dropna(subset=['last_two_digits'])
    
    values_origin = values
    
    # Filter by date range
    values['date_column'] = pd.to_datetime(values['date_column'], format='%d-%m-%Y')
    values['date_column_number'] = (values['date_column'] - pd.Timestamp("2002-01-01")) // pd.Timedelta('1D')
    values = values[(values['date_column'] >= start_date_filter_object) & (values['date_column'] <= end_date_filter_object)]
 
    # print(f"values: {values['last_two_digits'].values}")
    # Prepare the data and train the model as before
    
    # X = values[['date_column_number']].values
    # y = values['last_two_digits'].values
    
    # print(f"values[['date_column_number']].values: {values[['date_column_number']].values}")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f'Mean Squared Error: {mse}')
    
    # predicted = predict(model, values, start_date='01-01-2022', prediction_date='01-01-2023')
    # print(f"predict {predicted}")
    
    
        
    output_folder = "datalotto"
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Iterate through years in reverse order
    for year in range(2023, 2001, -1):
        start_date_betting = f'01-01-{year}'
        end_date_betting = f'31-12-{year}'

        result = calculate_statistics_with_kelly1(start_date_betting, end_date_betting, values, predict, get_lottery_numbers, values_origin)
        print(f"result {result}")
        
        df = pd.DataFrame(result)
        output_file = os.path.join(output_folder, f"data_{year}.csv")  # Create the output file path in the datalotto folder
        df.to_csv(output_file, index=False)
        
    # start_date_betting = '01-01-2017'
    # end_date_betting = '01-01-2018'
    # time_delta_days = 1
    # money_bet_per_day = 10
    # rate_of_return = 97
    # min_time_delta = 1
    # max_time_delta = 60
    # initial_bankroll = 100000

    # result = calculate_statistics_with_kelly1(start_date_betting, end_date_betting, initial_bankroll, rate_of_return, values, predict, get_lottery_numbers, values_origin)
    # print(f"result {result}")
     
    # df = pd.DataFrame(result)
    # df.to_csv(f"data_{start_date_betting}_{end_date_betting}.csv", index=False)
    
# Call the optimize_bet_size function to find the optimal bet size
   # optimal_bet_size = optimize_bet_size(values, start_date_betting, end_date_betting, initial_bankroll, rate_of_return, predict, get_lottery_numbers)
       # Use the optimal bet size in the calculate_statistics function
   # result = calculate_statistics(start_date_betting, end_date_betting, initial_bankroll, rate_of_return, optimal_bet_size, values, predict, get_lottery_numbers)
    # Use the optimal bet size in the calculate_statistics function
    #result = calculate_statistics(start_date_betting, end_date_betting, initial_bankroll, rate_of_return, optimal_bet_size, values, predict, get_lottery_numbers)

import numpy as np
import random
from functools import partial

fibonacci_sequence = [1, 1]
def get_fibonacci_number(n):
    if n < len(fibonacci_sequence):
        return fibonacci_sequence[n]
    else:
        next_fibonacci_number = get_fibonacci_number(n - 1) + get_fibonacci_number(n - 2)
        fibonacci_sequence.append(next_fibonacci_number)
        return next_fibonacci_number


statistical_model()

