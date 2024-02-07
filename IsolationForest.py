import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np


def main_function(IFdf):
    # Create and train an Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.5)  # Adjust the contamination parameter as needed
    isolation_forest.fit(IFdf[['Value']])

    # Predict anomalies using the trained model and add the anomaly scores to your DataFrame
    IFdf['Anomaly_Score'] = isolation_forest.decision_function(IFdf[['Value']])
    IFdf['Time'] = range(1, len(IFdf) + 1)

    # Define a threshold to classify anomalies
    # threshold_min = -0.35  # Adjust the threshold based on your data and requirements
    # threshold_max = 0.15
    threshold = -0.35
    # df['Is_Anomaly'] = (df['Anomaly_Score'] < threshold_min) | (df['Anomaly_Score'] > threshold_max)
    if (IFdf['Anomaly_Score'] < threshold):
        IFdf['IF : Anomaly'] = 0.0
    else:
        IFdf['IF : Anomaly'] = 1.0

    #plot(IFdf)
    #plot_compare(IFdf, threshold)
    # save_to_csv(df)
    return IFdf


def plot(df):
    # Visualize the anomaly scores
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Time'], df['Value'], label='Data points')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Scatter plot')
    plt.legend()
    plt.show()

    # Create a 2D histogram (hist2d) of Time vs Anomaly Score
    plt.figure(figsize=(10, 6))
    plt.hist2d(df['Time'], df['Value'], bins=(100, 100), cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.axhline(y=0.15, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-0.35, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection with Isolation Forest (2D Histogram)')
    plt.legend()
    plt.show()


def plot_compare(df, threshold):
    # Create a figure for the values vs. time
    fig1, axs1 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Value vs. Time and Anomaly Score vs. Time', fontsize='16')
    axs1[0].plot(df['Time'], df['Value'], label='Value', color='b')
    axs1[0].set_ylabel('Value', fontsize='12')
    axs1[0].set_title('Value vs. Time', fontsize='12')
    axs1[0].legend(loc='upper left')

    # Create a figure for the anomaly scores vs. time
    axs1[1].plot(df['Time'], df['Anomaly_Score'], label='Anomaly Score', color='b')
    axs1[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    # axs1[1].axhline(y=threshold_min, color='r', linestyle='--', label='Threshold (Min)')
    # axs1[1].axhline(y=threshold_max, color='g', linestyle='--', label='Threshold (Max)')
    axs1[1].set_xlabel('Time', fontsize='12')
    axs1[1].set_ylabel('Anomaly Score', fontsize='12')
    axs1[1].set_title('Anomaly Score vs. Time', fontsize='12')
    axs1[1].legend(loc='upper left')

    plt.show()


def save_to_csv(df):
    # Save the results to a CSV file
    output_filename = 'C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Isolation_Forest_results.csv'
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

