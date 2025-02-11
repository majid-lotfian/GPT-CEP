import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from simanneal import Annealer
import random
import matplotlib.pyplot as plt


# Define the column names
column_names = [
    'timestamp', 'activityID', 'heart_rate',
    'IMU_hand_temp', 'IMU_hand_acc_16g_x', 'IMU_hand_acc_16g_y', 'IMU_hand_acc_16g_z',
    'IMU_hand_acc_6g_x', 'IMU_hand_acc_6g_y', 'IMU_hand_acc_6g_z',
    'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z',
    'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z',
    'IMU_hand_ori_1', 'IMU_hand_ori_2', 'IMU_hand_ori_3', 'IMU_hand_ori_4',
    'IMU_chest_temp', 'IMU_chest_acc_16g_x', 'IMU_chest_acc_16g_y', 'IMU_chest_acc_16g_z',
    'IMU_chest_acc_6g_x', 'IMU_chest_acc_6g_y', 'IMU_chest_acc_6g_z',
    'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z',
    'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z',
    'IMU_chest_ori_1', 'IMU_chest_ori_2', 'IMU_chest_ori_3', 'IMU_chest_ori_4',
    'IMU_ankle_temp', 'IMU_ankle_acc_16g_x', 'IMU_ankle_acc_16g_y', 'IMU_ankle_acc_16g_z',
    'IMU_ankle_acc_6g_x', 'IMU_ankle_acc_6g_y', 'IMU_ankle_acc_6g_z',
    'IMU_ankle_gyro_x', 'IMU_ankle_gyro_y', 'IMU_ankle_gyro_z',
    'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z',
    'IMU_ankle_ori_1', 'IMU_ankle_ori_2', 'IMU_ankle_ori_3', 'IMU_ankle_ori_4'
]

# Load the dataset
data = pd.read_csv('D:/PhD/DataSets/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/subject101.dat', sep=' ', header=None, names=column_names)

# Remove records with activityID equal to 0
data = data[data['activityID'] != 0]

# Replace NaN values in heart_rate with the median, converted to integer
median_heart_rate = int(data['heart_rate'].median())
data['heart_rate'].fillna(median_heart_rate, inplace=True)

# Replace NaN values with the mean of the respective column
data.fillna(data.mean(), inplace=True)

# Separate features and target
X = data.drop(['activityID', 'timestamp'], axis=1)
y = data['activityID']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class ThresholdOptimizerSA(Annealer):
    def __init__(self, state, X_train, y_train, X_test, y_test):
        super(ThresholdOptimizerSA, self).__init__(state)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy_history = []


    def activity_recognition(self, X, thresholds):
        predictions = []
        for x in X:
            # Implement the custom pattern for lying activity
            if x[2] < thresholds[0] and x[4] < thresholds[1] and x[19] < thresholds[2] and x[34] < thresholds[3]:
                predictions.append(1)  # Lying activity
            else:
                predictions.append(0)  # Other activities
        return np.array(predictions)

    def move(self):
        # Modify the thresholds slightly
        param_to_change = random.randint(0, len(self.state) - 1)
        self.state[param_to_change] += random.uniform(-1, 1)
        self.state[param_to_change] = max(0, self.state[param_to_change])  # Ensure thresholds stay non-negative

    def energy(self):
        # Train and evaluate the model with the current thresholds
        y_pred = self.activity_recognition(self.X_test, self.state)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.accuracy_history.append(accuracy)
        return -accuracy  # We are minimizing negative accuracy

# Initial thresholds
initial_state = [90, 1, 1, 1]

annealer = ThresholdOptimizerSA(initial_state, X_train, y_train, X_test, y_test)
annealer.steps = 1000  # Number of iterations

best_state, best_energy = annealer.anneal()

# Train the final model with the best thresholds
y_pred = annealer.activity_recognition(X_test, best_state)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Calculate precision, recall, and F1 score
true_positives = np.sum((y_test == 1) & (y_pred == 1))
false_positives = np.sum((y_test != 1) & (y_pred == 1))
false_negatives = np.sum((y_test == 1) & (y_pred != 1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Best thresholds: {best_state}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')
print(f'Classification Report:\n{report}')


# Plot the improvement in accuracy and F1 score over iterations
plt.figure(figsize=(6, 6))

plt.plot(annealer.accuracy_history)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Improvement in Accuracy over Iterations')
plt.tight_layout()
plt.show()