import matplotlib.pyplot as plt
import numpy as np
import stumpy


def main_function(df):
    plot(df)
    matrix_algorithm(df)
    # save_to_csv(df)
    return df


def plot(df):
    plt.suptitle('Dataset', fontsize='30')
    plt.xlabel('Time', fontsize='20')
    plt.ylabel('Value', fontsize='20')
    plt.plot(df['Value'].values)
    plt.show()


def matrix_algorithm(df):
    m = 100
    mp = stumpy.stump(df['Value'], m)
    print("mp")

    discord_idx = np.argsort(mp[:, 0])[-1]

    print(f"The discord is located at index {discord_idx}")

    nearest_neighbor_distance = mp[discord_idx, 0]

    print(f"The nearest neighbor subsequence to this discord is {nearest_neighbor_distance} units away")

    anomaly_threshold = 5

    # Create an array to store whether each data point is an anomaly (1 for anomaly, 0 for normal)
    anomalies = np.where(mp[:, 0] > anomaly_threshold, 1, 0)

    # Ensure the lengths match
    if len(anomalies) == len(df):
        # Add Is_Anomaly column to DataFrame
        df['MP : Anomaly'] = [True if a == 1 else False for a in anomalies]
    else:
        # Create a list of False values with the same length as df
        is_anomaly_values = [False] * len(df)

        # Assign True only for corresponding values
        for i in range(len(anomalies)):
            if i < len(df):
                is_anomaly_values[i] = True if anomalies[i] == 1 else False

        # Add Is_Anomaly column to DataFrame
        df['MP : Anomaly'] = is_anomaly_values

    plot_compare(df, discord_idx, mp)


def plot_compare(df, discord_idx, mp):
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Discord (Anomaly) Discovery', fontsize='30')
    axs[0].plot(df['Value'].values)
    axs[0].set_ylabel('Value', fontsize='20')
    axs[1].set_xlabel('Time', fontsize='20')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=discord_idx, linestyle="dashed")
    plt.axhline(y=5, color='r', linestyle='--', label='Threshold')
    axs[1].plot(mp[:, 0])
    plt.show()


def save_to_csv(df):
    output_filename = 'C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Matrix_Profiling_results.csv'
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load the dataset
# Assuming df is your DataFrame with 'Time' and 'Label' columns
# Feature engineering can be performed if there are additional features
# Scaling might be necessary depending on your data

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Prepare data for training
X_train = train[['Time']]
y_train = train['Label']

# Prepare data for testing
X_test = test[['Time']]
y_test = test['Label']

# Create and train Isolation Forest model
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_train)

# Predict anomalies on the test set
y_pred = clf.predict(X_test)

# Convert predictions to 0 (normal) and 1 (anomaly)
y_pred[y_pred == 1] = 0  # 1 indicates normal, so convert to 0
y_pred[y_pred == -1] = 1  # -1 indicates anomaly, so convert to 1

# Evaluate the model
print(classification_report(y_test, y_pred))
