# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

# Load the dataset
data = pd.read_csv('/mnt/data/Alzheimer_Disease_Healthy_Aging_Data.csv')

# Display basic information about the dataset
print("Dataset Info:")
data.info()
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values with mean/median
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        data[column].fillna(data[column].mean(), inplace=True)
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Check the correlation between features and the target variable
print("\nCorrelation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix['Data_Value'].sort_values(ascending=False))

# Features and target
y = data['Data_Value']
X = data.drop(columns=['Data_Value', 'RowId'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Implementation

# Define the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
nn_eval = model.evaluate(X_test, y_test, verbose=0)
print("\nNeural Network Test Accuracy:", nn_eval[1])

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Random Forest (Ensemble Learning)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print("\nRandom Forest Test Accuracy:", rf_acc)

# XGBoost (Ensemble Learning)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print("\nXGBoost Test Accuracy:", xgb_acc)

# Visualization of model accuracies
model_names = ['Neural Network', 'Random Forest', 'XGBoost']
accuracies = [nn_eval[1], rf_acc, xgb_acc]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Comparison of Model Accuracies')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=10)
plt.show()

# Comparison of Results
print("\nComparison of Results:")
print(f"Neural Network Accuracy: {nn_eval[1]:.2f}")
print(f"Random Forest Accuracy: {rf_acc:.2f}")
print(f"XGBoost Accuracy: {xgb_acc:.2f}")

# Detailed classification report for the best model
best_model = None
if max(nn_eval[1], rf_acc, xgb_acc) == nn_eval[1]:
    best_model = 'Neural Network'
    best_preds = (model.predict(X_test) > 0.5).astype(int).flatten()
elif max(nn_eval[1], rf_acc, xgb_acc) == rf_acc:
    best_model = 'Random Forest'
    best_preds = rf_preds
else:
    best_model = 'XGBoost'
    best_preds = xgb_preds

print(f"\nDetailed Classification Report for {best_model}:")
print(classification_report(y_test, best_preds))
