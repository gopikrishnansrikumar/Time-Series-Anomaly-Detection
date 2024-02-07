import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.model_selection import train_test_split
import stumpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import csv
from sklearn.preprocessing import MinMaxScaler


# Read the data into a Pandas DataFrame
df = pd.read_csv('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/MBA_ECG805_data_12.out', header=None, nrows=10000)
df.columns = ['Value', 'Label']  # Assign column names as needed
df['Value'] = df['Value'].astype(float)  # Ensure the 'Value' column is of the float data type

# Create a sequence of time values starting from 1
df['Time'] = range(1, len(df) + 1)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)


# to present data set in graphical form
def visualize_data():
    # Plot 'Value' against 'Time' with conditional coloring
    plt.scatter(df['Time'], df['Value'], color='black', s=5)  # default color for points labeled as 0.0
    plt.ylabel('Value')
    plt.title('Dataset')
    plt.show()

    # Conditionally color points labeled as 1.0
    mask = df['Label'] == 0.0
    plt.scatter(df.loc[mask, 'Time'], df.loc[mask, 'Value'], color='black', label='Label 0.0', s=5)
    mask = df['Label'] == 1.0
    plt.scatter(df.loc[mask, 'Time'], df.loc[mask, 'Value'], color='red', label='Label 1.0', s=5)

    plt.ylabel('Value')
    plt.title('Dataset with anomalies')
    plt.legend()
    plt.show()


# Isolation Forest
def isolation_forest():
    # Train the Isolation Forest model on the training set
    isolation_forest_train = IsolationForest(contamination=0.1)  # Adjust the contamination parameter
    isolation_forest_train.fit(train_df[['Value']])

    # Predict anomalies on the training set
    train_predictions = isolation_forest_train.predict(train_df[['Value']])
    train_df['Anomaly Prediction IF'] = np.where(train_predictions == 1, 0.0, 1.0)

    # Get anomaly scores for each sample in the training set
    train_anomaly_scores = isolation_forest_train.decision_function(train_df[['Value']])
    train_df['AnomalyScore IF'] = train_anomaly_scores

    # Predict anomalies on the test set
    test_predictions = isolation_forest_train.predict(test_df[['Value']])
    test_df['Anomaly Prediction IF'] = np.where(test_predictions == 1, 0.0, 1.0)

    # Get anomaly scores for each sample in the test set
    test_anomaly_scores = isolation_forest_train.decision_function(test_df[['Value']])
    test_df['AnomalyScore IF'] = test_anomaly_scores

    # Plot for Training Set
    fig, axs_train = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Data vs Time for training set
    mask_label_1_train = train_df['Label'] == 1.0
    axs_train[0].scatter(train_df.loc[~mask_label_1_train, 'Time'], train_df.loc[~mask_label_1_train, 'Value'], color='black', label='Label 0.0', s=5)
    axs_train[0].scatter(train_df.loc[mask_label_1_train, 'Time'], train_df.loc[mask_label_1_train, 'Value'], color='red', label='Label 1.0', s=5)
    axs_train[0].set_title('Training Data vs Time')
    axs_train[0].set_ylabel('Value')
    axs_train[0].legend()

    # Plot 2: Anomaly Score vs Time for training set with connections between consecutive points
    threshold_train = np.percentile(train_df['AnomalyScore IF'], 100 * 0.1)  # Calculate the threshold
    mask_anomaly_0_train = train_df['Anomaly Prediction IF'] == 0.0
    mask_anomaly_1_train = train_df['Anomaly Prediction IF'] == 1.0

    axs_train[1].scatter(train_df.loc[mask_anomaly_0_train, 'Time'], train_df.loc[mask_anomaly_0_train, 'AnomalyScore IF'], color='black', s=5)
    axs_train[1].scatter(train_df.loc[mask_anomaly_1_train, 'Time'], train_df.loc[mask_anomaly_1_train, 'AnomalyScore IF'], color='red', s=5)
    axs_train[1].axhline(y=threshold_train, color='gray', linestyle='--', label='Threshold')
    axs_train[1].set_title('Training Anomaly Score vs Time')
    axs_train[1].set_ylabel('Anomaly Score')
    axs_train[1].legend()

    # Plot 3: Anomaly Prediction vs Time for training set
    mask_anomaly_1_train = train_df['Anomaly Prediction IF'] == 1.0
    axs_train[2].scatter(train_df.loc[~mask_anomaly_1_train, 'Time'], train_df.loc[~mask_anomaly_1_train, 'Value'], color='black', label='Not Anomaly', s=5)
    axs_train[2].scatter(train_df.loc[mask_anomaly_1_train, 'Time'], train_df.loc[mask_anomaly_1_train, 'Value'], color='red', label='Anomaly', s=5)
    axs_train[2].set_title('Training Anomaly Prediction vs Time')
    axs_train[2].set_ylabel('Value')
    axs_train[2].legend()

    plt.tight_layout()
    plt.show()

    # Plot for Test Set
    fig, axs_test = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Data vs Time for test set
    mask_label_1_test = test_df['Label'] == 1.0
    axs_test[0].scatter(test_df.loc[~mask_label_1_test, 'Time'], test_df.loc[~mask_label_1_test, 'Value'], color='black', label='Label 0.0', s=5)
    axs_test[0].scatter(test_df.loc[mask_label_1_test, 'Time'], test_df.loc[mask_label_1_test, 'Value'], color='red', label='Label 1.0', s=5)
    axs_test[0].set_title('Test Data vs Time')
    axs_test[0].set_ylabel('Value')
    axs_test[0].legend()

    # Plot 2: Anomaly Score vs Time for test set
    threshold_test = np.percentile(test_df['AnomalyScore IF'], 100 * 0.1)  # Calculate the threshold
    mask_anomaly_0_test = test_df['Anomaly Prediction IF'] == 0.0
    mask_anomaly_1_test = test_df['Anomaly Prediction IF'] == 1.0

    axs_test[1].scatter(test_df.loc[mask_anomaly_0_test, 'Time'], test_df.loc[mask_anomaly_0_test, 'AnomalyScore IF'], color='black', label='Anomaly Prediction 0.0', s=5)
    axs_test[1].scatter(test_df.loc[mask_anomaly_1_test, 'Time'], test_df.loc[mask_anomaly_1_test, 'AnomalyScore IF'], color='red', label='Anomaly Prediction 1.0', s=5)
    axs_test[1].axhline(y=threshold_test, color='gray', linestyle='--', label='Threshold')
    axs_test[1].set_title('Test Anomaly Score vs Time')
    axs_test[1].set_ylabel('Anomaly Score')
    axs_test[1].legend()

    # Plot 3: Anomaly Prediction vs Time for test set
    mask_anomaly_1_test = test_df['Anomaly Prediction IF'] == 1.0
    axs_test[2].scatter(test_df.loc[~mask_anomaly_1_test, 'Time'], test_df.loc[~mask_anomaly_1_test, 'Value'], color='black', label='Anomaly Prediction 0.0', s=5)
    axs_test[2].scatter(test_df.loc[mask_anomaly_1_test, 'Time'], test_df.loc[mask_anomaly_1_test, 'Value'], color='red', label='Anomaly Prediction 1.0', s=5)
    axs_test[2].set_title('Test Anomaly Prediction vs Time')
    axs_test[2].set_ylabel('Value')
    axs_test[2].legend()

    plt.tight_layout()
    plt.show()

    # Select the desired columns
    selected_columns = test_df[['Label', 'Anomaly Prediction IF']]

    # Save to a CSV file
    selected_columns.to_csv('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_IF.csv', index=False)

    '''result_df = pd.DataFrame()
    result_df['Label'] = test_df['Label']
    result_df['Anomaly Prediction IF'] = test_df['Anomaly Prediction IF']
    return result_df['Label', 'Anomaly Prediction IF']'''


def matrix_profile():

    m = 100
    mp = stumpy.stump(df['Value'], m)

    discord_idx = np.argsort(mp[:, 0])[-1]

    nearest_neighbor_distance = mp[discord_idx, 0]

    anomaly_threshold = 5

    # Create an array to store whether each data point is an anomaly (1 for anomaly, 0 for normal)
    anomalies = np.where(mp[:, 0] > anomaly_threshold, 1.0, 0.0)

    # Ensure the lengths match
    if len(anomalies) == len(df):
        # Add Is_Anomaly column to DataFrame
        df['Anomaly Prediction MP'] = [1.0 if a == 1 else 0.0 for a in anomalies]
    else:
        # Create a list of False values with the same length as df
        is_anomaly_values = [0.0] * len(df)

        # Assign True only for corresponding values
        for i in range(len(anomalies)):
            if i < len(df):
                is_anomaly_values[i] = 1.0 if anomalies[i] == 1 else 0.0

        # Add Is_Anomaly column to DataFrame
        df['Anomaly Prediction MP'] = is_anomaly_values

    print(df.head())

    # Plot for Matrix Profiling
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Data vs Time
    mask_label_1 = df['Label'] == 1.0
    axs[0].scatter(df.loc[~mask_label_1, 'Time'], df.loc[~mask_label_1, 'Value'], color='black', label='Label 0.0', s=5)
    axs[0].scatter(df.loc[mask_label_1, 'Time'], df.loc[mask_label_1, 'Value'], color='red', label='Label 1.0', s=5)
    axs[0].set_title('Data vs Time')
    axs[0].set_ylabel('Value')
    axs[0].legend()

    # Plot 2: Matrix Profile vs Time
    threshold_test = 5
    axs[1].plot(mp[:, 0], color='red')
    axs[1].axvline(x=discord_idx, linestyle="dashed", color='red', label='Discord')
    axs[1].axhline(y=threshold_test, color='gray', linestyle='--', label='Threshold')
    axs[1].set_title('Matrix Profile vs Time')
    axs[1].set_ylabel('Matrix Profile')
    axs[1].legend()

    # Plot 3: Anomaly Prediction vs Time
    mask_anomaly_1 = df['Anomaly Prediction MP'] == 1.0
    axs[2].scatter(df.loc[~mask_anomaly_1, 'Time'], df.loc[~mask_anomaly_1, 'Value'], color='black', label='Not Anomaly', s=5)
    axs[2].scatter(df.loc[mask_anomaly_1, 'Time'], df.loc[mask_anomaly_1, 'Value'], color='red', label='Anomaly', s=5)
    axs[2].set_title('Anomaly Prediction vs Time')
    axs[2].set_ylabel('Value')
    axs[2].legend()

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

    # Select the desired columns
    selected_columns = df[['Label', 'Anomaly Prediction MP']]

    # Save to a CSV file
    selected_columns.to_csv('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_MP.csv', index=False)    
        
    '''result_df = pd.DataFrame()
    result_df['Label'] = df['Label']
    result_df['Anomaly Prediction MP'] = df['Anomaly Prediction MP']
    return result_df['Label', 'Anomaly Prediction MP']'''


def knn_method():
    # Train the KNN model on the training set
    knn_model = LocalOutlierFactor(n_neighbors=10, contamination=0.1)  # Adjust parameters as needed
    knn_model.fit(train_df[['Value']])

    # Predict anomalies on the training set
    train_predictions = knn_model.fit_predict(train_df[['Value']])
    train_df['Anomaly Prediction KNN'] = np.where(train_predictions == -1, 1.0, 0.0)

    # Get anomaly scores for each sample in the training set
    train_anomaly_scores = -knn_model.negative_outlier_factor_
    train_df['AnomalyScore KNN'] = train_anomaly_scores

    # Predict anomalies on the test set
    test_predictions = knn_model.fit_predict(test_df[['Value']])
    test_df['Anomaly Prediction KNN'] = np.where(test_predictions == -1, 1.0, 0.0)

    # Get anomaly scores for each sample in the test set
    test_anomaly_scores = -knn_model.negative_outlier_factor_
    test_df['AnomalyScore KNN'] = test_anomaly_scores

    # Plot for Training Set
    fig, axs_train = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Data vs Time for training set
    mask_label_1_train = train_df['Label'] == 1.0
    axs_train[0].scatter(train_df.loc[~mask_label_1_train, 'Time'], train_df.loc[~mask_label_1_train, 'Value'], color='black', label='Label 0.0', s=5)
    axs_train[0].scatter(train_df.loc[mask_label_1_train, 'Time'], train_df.loc[mask_label_1_train, 'Value'], color='red', label='Label 1.0', s=5)
    axs_train[0].set_title('Training Data vs Time')
    axs_train[0].set_ylabel('Value')
    axs_train[0].legend()

    # Plot 2: Anomaly Score vs Time for training set with connections between consecutive points
    mask_anomaly_0_train_knn = train_df['Anomaly Prediction KNN'] == 0.0
    mask_anomaly_1_train_knn = train_df['Anomaly Prediction KNN'] == 1.0

    axs_train[1].scatter(train_df.loc[mask_anomaly_0_train_knn, 'Time'], train_df.loc[mask_anomaly_0_train_knn, 'AnomalyScore KNN'], color='black', s=5)
    axs_train[1].scatter(train_df.loc[mask_anomaly_1_train_knn, 'Time'], train_df.loc[mask_anomaly_1_train_knn, 'AnomalyScore KNN'], color='red', s=5)
    axs_train[1].set_title('Training Anomaly Score vs Time (KNN)')
    axs_train[1].set_ylabel('Anomaly Score (KNN)')

    # Plot 3: Anomaly Prediction vs Time for training set
    mask_anomaly_1_train_knn = train_df['Anomaly Prediction KNN'] == 1.0
    axs_train[2].scatter(train_df.loc[~mask_anomaly_1_train_knn, 'Time'], train_df.loc[~mask_anomaly_1_train_knn, 'Value'], color='black', label='Not Anomaly', s=5)
    axs_train[2].scatter(train_df.loc[mask_anomaly_1_train_knn, 'Time'], train_df.loc[mask_anomaly_1_train_knn, 'Value'], color='red', label='Anomaly', s=5)
    axs_train[2].set_title('Training Anomaly Prediction vs Time (KNN)')
    axs_train[2].set_ylabel('Value')
    axs_train[2].legend()

    plt.tight_layout()
    plt.show()

    # Plot for Test Set
    fig, axs_test = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Data vs Time for test set
    mask_label_1_test_knn = test_df['Label'] == 1.0
    axs_test[0].scatter(test_df.loc[~mask_label_1_test_knn, 'Time'], test_df.loc[~mask_label_1_test_knn, 'Value'], color='black', label='Label 0.0', s=5)
    axs_test[0].scatter(test_df.loc[mask_label_1_test_knn, 'Time'], test_df.loc[mask_label_1_test_knn, 'Value'], color='red', label='Label 1.0', s=5)
    axs_test[0].set_title('Test Data vs Time')
    axs_test[0].set_ylabel('Value')
    axs_test[0].legend()

    # Plot 2: Anomaly Score vs Time for test set
    mask_anomaly_0_test_knn = test_df['Anomaly Prediction KNN'] == 0.0
    mask_anomaly_1_test_knn = test_df['Anomaly Prediction KNN'] == 1.0

    axs_test[1].scatter(test_df.loc[mask_anomaly_0_test_knn, 'Time'], test_df.loc[mask_anomaly_0_test_knn, 'AnomalyScore KNN'], color='black', s=5)
    axs_test[1].scatter(test_df.loc[mask_anomaly_1_test_knn, 'Time'], test_df.loc[mask_anomaly_1_test_knn, 'AnomalyScore KNN'], color='red', s=5)
    axs_test[1].set_title('Test Anomaly Score vs Time (KNN)')
    axs_test[1].set_ylabel('Anomaly Score (KNN)')

    # Plot 3: Anomaly Prediction vs Time for test set
    mask_anomaly_1_test_knn = test_df['Anomaly Prediction KNN'] == 1.0
    axs_test[2].scatter(test_df.loc[~mask_anomaly_1_test_knn, 'Time'], test_df.loc[~mask_anomaly_1_test_knn, 'Value'], color='black', label='Not Anomaly', s=5)
    axs_test[2].scatter(test_df.loc[mask_anomaly_1_test_knn, 'Time'], test_df.loc[mask_anomaly_1_test_knn, 'Value'], color='red', label='Anomaly', s=5)
    axs_test[2].set_title('Test Anomaly Prediction vs Time (KNN)')
    axs_test[2].set_ylabel('Value')
    axs_test[2].legend()

    plt.tight_layout()
    plt.show()

    '''result_df = pd.DataFrame()
    result_df['Label'] = test_df['Label']
    result_df['Anomaly Prediction KNN'] = test_df['Anomaly Prediction KNN']
    return result_df['Label', 'Anomaly Prediction KNN']'''
    
    # Select the desired columns
    selected_columns = test_df[['Label', 'Anomaly Prediction KNN']]

    # Save to a CSV file
    selected_columns.to_csv('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_KNN.csv', index=False)


def compare(file_path):
    df = pd.read_csv(file_path)

    # Assuming the first column is ground truth label and the second column is predicted label
    ground_truth = df.iloc[:, 0]
    predicted_labels = df.iloc[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(ground_truth, predicted_labels)
    precision = precision_score(ground_truth, predicted_labels)
    recall = recall_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_labels).ravel()

    # Area Under the ROC Curve
    roc_auc = roc_auc_score(ground_truth, predicted_labels)

    # Print or return the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # Return the metrics if needed
    return accuracy, precision, recall, f1, tp, fp, fn, tn, roc_auc


def compare_IF_KNN():

    cf = pd.read_csv('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/MBA_ECG805_data_12.out', header=None, nrows=3000)
    cf.columns = ['Value', 'Label']  # Assign column names as needed
    cf['Value'] = cf['Value'].astype(float)  # Ensure the 'Value' column is of the float data type

    # Create a sequence of time values starting from 1
    cf['Time'] = range(1, len(cf) + 1)

    # Train the Isolation Forest model on the training set
    isolation_forest_train = IsolationForest(contamination=0.1)  # Adjust the contamination parameter
    isolation_forest_train.fit(cf[['Value']])

    # Predict anomalies on the training set
    train_predictions = isolation_forest_train.predict(cf[['Value']])
    cf['Anomaly Prediction IF'] = np.where(train_predictions == 1, 0.0, 1.0)

    # Get anomaly scores for each sample in the training set
    train_anomaly_scores = isolation_forest_train.decision_function(cf[['Value']])
    cf['AnomalyScore IF'] = train_anomaly_scores

    knn_model = LocalOutlierFactor(n_neighbors=10, contamination=0.1)  # Adjust parameters as needed
    knn_model.fit(cf[['Value']])

    # Predict anomalies on the training set
    train_predictions = knn_model.fit_predict(cf[['Value']])
    cf['Anomaly Prediction KNN'] = np.where(train_predictions == -1, 1.0, 0.0)

    # Get anomaly scores for each sample in the training set
    train_anomaly_scores = -knn_model.negative_outlier_factor_
    cf['AnomalyScore KNN'] = np.abs(train_anomaly_scores)

    print(cf['AnomalyScore IF'])
    print(cf['AnomalyScore KNN'])

    cf['AnomalyScore IF'] = 1-((cf['AnomalyScore IF']-cf['AnomalyScore IF'].min())/(cf['AnomalyScore IF'].max()-cf['AnomalyScore IF'].min()))
    cf['AnomalyScore KNN'] = (cf['AnomalyScore KNN']-cf['AnomalyScore KNN'].min())/(cf['AnomalyScore KNN'].max()-cf['AnomalyScore KNN'].min())

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Subplot 1: Data vs Time for test set
    mask_label_1_test = cf['Label'] == 1.0
    axs[0].scatter(cf.loc[~mask_label_1_test, 'Time'], cf.loc[~mask_label_1_test, 'Value'], color='black', label='Label 0.0', s=5)
    axs[0].scatter(cf.loc[mask_label_1_test, 'Time'], cf.loc[mask_label_1_test, 'Value'], color='red', label='Label 1.0', s=5)
    axs[0].set_title('Test Data vs Time')
    axs[0].set_ylabel('Value')
    axs[0].legend()

    # Subplot 2: Anomaly Scores vs Time for test set

    axs[1].step(cf['Time'], cf['AnomalyScore IF'], label='Isolation Forest', color='blue', alpha=0.9)
    bar_width = 25
    axs[1].bar(cf['Time'], cf['AnomalyScore KNN'], width=bar_width, label='KNN', color='red', alpha=0.5)  # Adjust alpha for transparency
    axs[1].set_ylabel('Anomaly score')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


visualize_data()
isolation_forest()
matrix_profile()
knn_method()
compare_IF_KNN()


print("Isolation Forest")
IF_metrics = compare('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_IF.csv')
print("Matrix Profiling")
MP_metrics = compare('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_MP.csv')
print("k Nearest Neighbors")
KNN_metrix = compare('C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_KNN.csv')

# List of headers
headers = ["Accuracy", "Precision", "Recall", "F1 Score", "True Positives", "False Positives", "False Negatives", "True Negatives", "AUC-ROC"]

# List of tuples
data = [IF_metrics, MP_metrics, KNN_metrix]

# CSV file path
csv_file_path = "C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/Anomaly_metrics.csv"

# Writing to CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write headers
    writer.writerow(["Algorithm"] + headers)

    # Write data
    writer.writerow(["Isolation Forest"] + list(IF_metrics))
    writer.writerow(["Matrix Profiling"] + list(MP_metrics))
    writer.writerow(["k Nearest Neighbors"] + list(KNN_metrix))

print(f"CSV file saved at: {csv_file_path}")


