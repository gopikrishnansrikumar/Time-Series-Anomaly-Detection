# Time-Series-Anomaly-Detection
The process of identifying and labelling anomalies in a time series is called time series anomaly detection. Different anomaly detection techniques use contrasting approaches for marking irregularities. 

# Benchmark Algorithms

# Isolation Forest
Isolation Forest is an unsupervised machine learning algorithm designed for anomaly detection. Developed on the principle of isolating anomalies rather than normal data points, the method employs binary decision trees to create a model. In the training phase, the algorithm randomly selects a feature and a split value to isolate instances containing anomalies efficiently. Anomalies are expected to require fewer splits to be isolated in the tree structure, making them stand out. During the testing phase, the isolation forest can rapidly identify anomalies by measuring the average path length of instances through the trees. This approach is particularly effective for detecting outliers or rare events in large datasets and demonstrates efficiency in terms of both computational resources and accuracy.

![image](https://github.com/gopikrishnansrikumar/Time-Series-Anomaly-Detection/assets/138595672/644f5215-2a5f-46eb-b5ff-891814a13fac)

# One Class SVM (OCSVM)
One-Class Support Vector Machine (One-Class SVM) is a machine learning algorithm primarily used for anomaly detection. Unlike traditional SVMs designed for binary classification, One-Class SVM focuses on learning a decision boundary that encapsulates the majority of normal instances in the training data. It aims to create a hyperplane that separates the normal data from the rest of the feature space, making it particularly useful in scenarios where anomalies are scarce or difficult to obtain for training. The algorithm works by maximizing the margin around the normal data points, allowing it to generalize well to identify anomalies during testing. One-Class SVM is advantageous in situations where the majority of the data belongs to one class, making it suitable for applications such as fraud detection, fault diagnosis, or outlier detection in various domains.

![image](https://github.com/gopikrishnansrikumar/Time-Series-Anomaly-Detection/assets/138595672/a356be3b-bd3b-4242-b3d6-6242cb563e52)

# Local Outlier Factor (LOF)
The Local Outlier Factor (LOF) is an unsupervised machine learning algorithm designed for anomaly detection in datasets. Developed to identify local deviations of data points from their neighbors, LOF assigns an anomaly score to each instance based on its relative density compared to its surrounding data points. The algorithm considers the local density of a data point in relation to the densities of its neighbors, with anomalies exhibiting lower local densities than their neighbors receiving higher LOF scores. This approach allows LOF to effectively capture outliers that may not be distinguishable in a global context. LOF is particularly useful in scenarios where anomalies have varying densities and spatial distributions, making it versatile for applications such as fraud detection, network security, and quality control in industrial processes.

![image](https://github.com/gopikrishnansrikumar/Time-Series-Anomaly-Detection/assets/138595672/f584ee6d-ae21-4175-8e68-3d83e4244f31)



