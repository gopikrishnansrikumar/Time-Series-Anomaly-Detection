import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Load the training dataset
file_path_train = 'C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/MBA_ECG805_data_12_train.out'
column_names_train = ['Data', 'Label']
df_train = pd.read_csv(file_path_train, header=None, names=column_names_train)

# Load the test dataset
file_path_test = 'C:/Users/gksi9/PycharmProjects/dataset/venv/TSB-UAD/data/synthetic/MBA_ECG805_data_12_test.out'
column_names_test = ['Data', 'Label']
df_test = pd.read_csv(file_path_test, header=None, names=column_names_test)

# Convert labels to 1 and 0
df_train['Label'] = df_train['Label'].astype(int)
df_test['Label'] = df_test['Label'].astype(int)

# Add a time column starting from 1 for both training and test sets
df_train['Time'] = range(1, len(df_train) + 1)
df_test['Time'] = range(1, len(df_test) + 1)

# Separate features and labels for training set
X_train = df_train[['Data']]
y_train = df_train['Label']

# Separate features and labels for test set
X_test = df_test[['Data']]
y_test = df_test['Label']

# Train the Isolation Forest model on the training set
isolation_forest = IsolationForest(contamination=0.1)  # Adjust the contamination parameter
isolation_forest.fit(X_train)

# Predict anomalies on the training set
train_predictions = isolation_forest.predict(X_train)
df_train['AnomalyPrediction'] = train_predictions

# Predict anomalies on the test set
test_predictions = isolation_forest.predict(X_test)
df_test['AnomalyPrediction'] = test_predictions

print(y_test.head())
print(df_test.head())

# Calculate accuracy for the test set
accuracy = accuracy_score(y_test, test_predictions)

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Plotting for test set
plt.figure(figsize=(12, 8))

# Plot the time series data with actual labels for test set
plt.subplot(2, 1, 1)
plt.plot(df_test['Time'], df_test['Data'], color='blue', label='Data')
plt.plot(df_test['Time'][df_test['Label'] == 1], df_test['Data'][df_test['Label'] == 1], color='red', label='Anomalies')
plt.title('Test Data vs Time with Actual Labels')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()

# Plot the time series data with anomaly predictions for test set
plt.subplot(2, 1, 2)
plt.plot(df_test['Time'], df_test['Data'], color='blue', label='Data')
plt.plot(df_test['Time'][df_test['AnomalyPrediction'] == -1], df_test['Data'][df_test['AnomalyPrediction'] == -1], color='red', label='Anomalies')
plt.title('Test Data vs Time with Isolation Forest Anomaly Predictions')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()

plt.tight_layout()
plt.show()
