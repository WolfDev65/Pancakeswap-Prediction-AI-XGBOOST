import csv
from pcs_contract_fun import get_round_data_round_with_titles
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go


def get_unique_round_numbers(data):
    unique_rounds = set()
    for row in data:
        unique_rounds.add(int(row['Round']))
    return unique_rounds


def fill_csv_with_actual_data(csv_file):
    data = []

    with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Get unique round numbers
    unique_round_numbers = get_unique_round_numbers(data)

    # Retrieve patterns and epochs for unique round numbers
    patterns = {}
    epochs = {}
    for round_number in tqdm(unique_round_numbers, desc="Retrieving patterns"):
        pattern_dict = get_round_data_round_with_titles(round_number)
        # Assuming the dictionary has a 'Pattern' key
        patterns[round_number] = pattern_dict['Pattern']
        # Assuming the dictionary has an 'Epoch' key
        epochs[round_number] = pattern_dict['Epoch']

    # Fill Actual column with the patterns based on the round number and verify the epoch
    updated_data = []
    for row in data:
        round_number = int(row['Round'])
        if not row['Actual']:
            if round_number in patterns and round_number in epochs:
                row['Actual'] = patterns[round_number]
                # You can print or log a message here if you want to verify the epoch
                # print(f"Round: {round_number}, Epoch: {epochs[round_number]}")
        updated_data.append(row)

    # Update CSV file with filled data
    with open(csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Round', 'Prediction',
                      'Bull Confidence', 'Bear Confidence', 'Actual']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_data:
            writer.writerow(row)


def compare_predictions_and_actual(csv_file):
    data = []

    with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    correct_predictions = 0
    total_predictions = len(data)

    for row in data:
        if row['Prediction'] == row['Actual']:
            correct_predictions += 1

    incorrect_predictions = total_predictions - correct_predictions

    # Display results in a bar chart
    labels = ['Correct', 'Incorrect']
    values = [correct_predictions, incorrect_predictions]
    bars = plt.bar(labels, values)
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Prediction vs Actual Comparison')

    # Add annotations to the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.show()



def check_predictions(csv_file):
    df = pd.read_csv(csv_file)
    df['correct'] = df['Prediction'] == df['Actual']
    colors = ['red' if not c else 'green' for c in df['correct']]
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='grey',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color=[colors],
                   align='left'))
    ])
    fig.show()


# csv_file = 'predictions_rf.csv'
# csv_file = 'predictions_rfi1.csv'
csv_file = 'predictions_rfi2.csv'

def runBacktest():

    # Replace 'predictions.csv' with the name of the CSV file you want to process
    fill_csv_with_actual_data(csv_file)


def getResult():

    # Replace 'predictions.csv' with the name of the CSV file you want to process
    compare_predictions_and_actual(csv_file)


def patterncheck():
    # Replace 'predictions.csv' with the name of the CSV file you want to process
    check_predictions(csv_file)


runBacktest()

getResult()

patterncheck()
