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
        self.f1_history = []
        self.best_accuracy = -np.inf
        self.best_f1 = -np.inf
        self.best_thresholds = None

    def activity_recognition(self, X, thresholds):
        predictions = []
        for x in X:
            # Implement the custom pattern for Standing activity
            if x[2] < thresholds[0] and x[4] < thresholds[1] and x[19] < thresholds[2] and x[34] < thresholds[3]:
                predictions.append(3)  # Standing activity
            else:
                predictions.append(0)  # Other activities
        return np.array(predictions)

    def move(self):
        # Modify the thresholds slightly with a larger range
        param_to_change = random.randint(0, len(self.state) - 1)
        self.state[param_to_change] += random.uniform(-10, 10)
        self.state[param_to_change] = max(0, self.state[param_to_change])  # Ensure thresholds stay non-negative

    def energy(self):
        # Train and evaluate the model with the current thresholds
        y_pred = self.activity_recognition(self.X_test, self.state)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        true_positives = np.sum((self.y_test == 3) & (y_pred == 3))
        false_positives = np.sum((self.y_test != 3) & (y_pred == 3))
        false_negatives = np.sum((self.y_test == 3) & (y_pred != 3))
        
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        self.accuracy_history.append(accuracy)
        self.f1_history.append(f1)
        
        # Track the best thresholds found
        if f1 > self.best_f1:
            self.best_accuracy = accuracy
            self.best_f1 = f1
            self.best_thresholds = self.state.copy()
        
        return -f1  # We are minimizing negative F1 score

# Initial thresholds for Standing activity
initial_state = [110, 5, 5, 5]

# Create an instance of the ThresholdOptimizerSA class
annealer = ThresholdOptimizerSA(initial_state, X_train, y_train, X_test, y_test)
annealer.steps = 1000  # Number of iterations
annealer.Tmax = 10.0  # Adjust the initial temperature for better exploration
annealer.Tmin = 0.1   # Adjust the final temperature for better convergence

# Find the best thresholds using simulated annealing
best_state, best_energy = annealer.anneal()

# Evaluate the initial thresholds
initial_thresholds_pred = annealer.activity_recognition(X_test, initial_state)

# Calculate precision, recall, and F1 score for initial thresholds
initial_true_positives = np.sum((y_test == 3) & (initial_thresholds_pred == 3))
initial_false_positives = np.sum((y_test != 3) & (initial_thresholds_pred == 3))
initial_false_negatives = np.sum((y_test == 3) & (initial_thresholds_pred != 3))

initial_precision = initial_true_positives / (initial_true_positives + initial_false_positives + 1e-9)
initial_recall = initial_true_positives / (initial_true_positives + initial_false_negatives + 1e-9)
initial_f1_score = 2 * (initial_precision * initial_recall) / (initial_precision + initial_recall + 1e-9)

print("Initial Thresholds:")
print(f'Precision: {initial_precision}')
print(f'Recall: {initial_recall}')
print(f'F1 Score: {initial_f1_score}')

# Train the final model with the best thresholds
best_thresholds_pred = annealer.activity_recognition(X_test, best_state)

# Calculate precision, recall, and F1 score for best thresholds found
best_true_positives = np.sum((y_test == 3) & (best_thresholds_pred == 3))
best_false_positives = np.sum((y_test != 3) & (best_thresholds_pred == 3))
best_false_negatives = np.sum((y_test == 3) & (best_thresholds_pred != 3))

best_precision = best_true_positives / (best_true_positives + best_false_positives + 1e-9)
best_recall = best_true_positives / (best_true_positives + best_false_negatives + 1e-9)
best_f1_score = 2 * (best_precision * best_recall) / (best_precision + best_recall + 1e-9)

print("\nBest Thresholds Found:")
print(f'Best thresholds: {best_state}')
print(f'Accuracy: {annealer.best_accuracy}')
print(f'Precision: {best_precision}')
print(f'Recall: {best_recall}')
print(f'F1 Score: {best_f1_score}')
print(f'Classification Report for Standing Activity:\n{classification_report(y_test, best_thresholds_pred)}')

# Plot the improvement in accuracy and F1 score over iterations
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(annealer.accuracy_history)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Improvement in Accuracy over Iterations (Standing Activity)')

plt.subplot(1, 2, 2)
plt.plot(annealer.f1_history)
plt.xlabel('Iteration')
plt.ylabel('F1 Score')
plt.title('Improvement in F1 Score over Iterations (Standing Activity)')

plt.tight_layout()
plt.show()
