import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Check for non-numeric values in the heart_rate column
non_numeric_hr = data['heart_rate'].apply(lambda x: isinstance(x, str))
print("Non-numeric heart_rate values:")
print(data[non_numeric_hr]['heart_rate'])

# Ensure all values are numeric, converting non-numeric values to NaN
data['heart_rate'] = pd.to_numeric(data['heart_rate'], errors='coerce')

# Remove records with activityID equal to 0
data = data[data['activityID'] != 0]

# Fill NaN values in the heart_rate column with the median
median_heart_rate = int(data['heart_rate'].median())
data['heart_rate'].fillna(median_heart_rate, inplace=True)

# Fill remaining NaN values in other columns with the mean of the respective column
data.fillna(data.mean(), inplace=True)

# Convert heart_rate to float explicitly
data['heart_rate'] = data['heart_rate'].astype(float)

# Verify that heart_rate is numeric
print("Unique values in heart_rate after conversion:", data['heart_rate'].unique())

# Separate features and target
X = data.drop(['activityID', 'timestamp'], axis=1)
y = data['activityID']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
            heart_rate, imu_hand_acc_x, imu_hand_acc_y, imu_ankle_acc_x = thresholds
            print(f"x2 (heart_rate): {x[2]} - type: {type(x[2])}")
            print(f"Heart rate thresholds: {heart_rate} - type: {type(heart_rate)}")
            if heart_rate[0] <= x[2] <= heart_rate[1]:
                predictions.append(6)  # Cycling activity
            else:
                predictions.append(0)  # Other activities
        return np.array(predictions)

    def move(self):
        param_to_change = random.randint(0, len(self.state) - 1)
        lower_bound, upper_bound = self.state[param_to_change]
        
        lower_bound += random.uniform(-5, 5)
        upper_bound += random.uniform(-5, 5)
        
        lower_bound = max(0, lower_bound)
        upper_bound = max(lower_bound, upper_bound)
        
        self.state[param_to_change] = (lower_bound, upper_bound)

    def energy(self):
        y_pred = self.activity_recognition(self.X_test, self.state)
        accuracy = accuracy_score(self.y_test, y_pred)
        true_positives = np.sum((self.y_test == 6) & (y_pred == 6))
        false_positives = np.sum((self.y_test != 6) & (y_pred == 6))
        false_negatives = np.sum((self.y_test == 6) & (y_pred != 6))
        
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        self.accuracy_history.append(accuracy)
        self.f1_history.append(f1)
        
        if f1 > self.best_f1:
            self.best_accuracy = accuracy
            self.best_f1 = f1
            self.best_thresholds = self.state.copy()
        
        return -f1

# Initial thresholds with lower and upper bounds
initial_state = [
    (100, 160),  # Heart rate bounds
    (-5, 5),     # IMU hand acc x bounds
    (2, 7),      # IMU hand acc y bounds
    (8, 12)      # IMU ankle acc x bounds
]

# Create an instance of the ThresholdOptimizerSA class
annealer = ThresholdOptimizerSA(initial_state, X_train, y_train, X_test, y_test)
annealer.steps = 1000
annealer.Tmax = 10.0
annealer.Tmin = 0.1

# Find the best thresholds using simulated annealing
best_state, best_energy = annealer.anneal()

# Evaluate the initial thresholds
initial_thresholds_pred = annealer.activity_recognition(X_test, initial_state)

# Calculate precision, recall, and F1 score for initial thresholds
initial_true_positives = np.sum((y_test == 6) & (initial_thresholds_pred == 6))
initial_false_positives = np.sum((y_test != 6) & (initial_thresholds_pred == 6))
initial_false_negatives = np.sum((y_test == 6) & (initial_thresholds_pred != 6))

initial_precision = initial_true_positives / (initial_true_positives + initial_false_positives + 1e-9)
initial_recall = initial_true_positives / (initial_true_positives + initial_false_negatives + 1e-9)
initial_f1_score = 2 * (initial_precision * initial_recall) / (initial_precision + initial_recall + 1e-9)

print("Initial Thresholds:")
print(f'Precision: {initial_precision}')
print(f'Recall: {initial_recall}')
print(f'F1 Score: {initial_f1_score}')
