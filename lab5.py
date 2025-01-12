import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error



# Load the dataset
file_path = 'C:/Users/PC/Desktop/LABS/lab 5/synthetic_traffic_data - synthetic_traffic_data.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

traffic_data = pd.read_csv(file_path)

# Step 1: Check for Outliers
# Visualize distributions of numeric columns for outlier detection
numeric_columns = ['traffic_flow', 'temperature', 'humidity']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(numeric_columns):
    sns.boxplot(data=traffic_data, y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

# Step 2: Time-Series Data Handling
# Convert `timestamp` to datetime and sort the data
traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
traffic_data = traffic_data.sort_values(by='timestamp')

# Extract time-based features
traffic_data['hour'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek

# Display the first few rows after adding features
print(traffic_data[['timestamp', 'hour', 'day_of_week']].head())

# Load the weather dataset
weather_file_path = 'C:/Users/PC/Desktop/LABS/lab 5/synthetic_traffic_data - synthetic_traffic_data.csv'  # Replace with actual path
weather_data = pd.read_csv(weather_file_path)

# Convert `timestamp` in weather data to datetime format
weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

# Ensure weather data is sorted by timestamp
weather_data = weather_data.sort_values(by='timestamp')

# Merge traffic data with weather data on `timestamp`
merged_data = pd.merge(traffic_data, weather_data, on='timestamp', how='inner')

# Display the first few rows of the merged dataset
print(merged_data.head())

# Extract time-based features
traffic_data['hour'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek
traffic_data['is_weekend'] = traffic_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
traffic_data['is_rush_hour'] = traffic_data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

# Preview time-based features
print(traffic_data[['timestamp', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].head())

#Normalize weather data using Min-Max Scaling
scaler = MinMaxScaler()

# Scale columns: temperature, humidity, rain
weather_columns = ['temperature', 'humidity', 'rain']
traffic_data[weather_columns] = scaler.fit_transform(traffic_data[weather_columns])

# Handle missing values (if any exist) by filling with median
traffic_data[weather_columns] = traffic_data[weather_columns].fillna(traffic_data[weather_columns].median())

# Check the scaled weather data
print(traffic_data[weather_columns].head())

# Create lag features for traffic_flow
traffic_data['traffic_flow_lag_1h'] = traffic_data['traffic_flow'].shift(1)
traffic_data['traffic_flow_lag_1d'] = traffic_data['traffic_flow'].shift(24)  # Assuming data is hourly

# Handle missing values in lag features by filling with 0
traffic_data[['traffic_flow_lag_1h', 'traffic_flow_lag_1d']] = traffic_data[['traffic_flow_lag_1h', 'traffic_flow_lag_1d']].fillna(0)

# Preview lag features
print(traffic_data[['traffic_flow', 'traffic_flow_lag_1h', 'traffic_flow_lag_1d']].head())

# Standardize traffic flow

traffic_flow_scaler = StandardScaler()
traffic_data['traffic_flow_scaled'] = traffic_flow_scaler.fit_transform(traffic_data[['traffic_flow']])

# Check scaled traffic flow
print(traffic_data[['traffic_flow', 'traffic_flow_scaled']].head())

# Step 2.1: Train-Test Split

# Define feature columns and target variable
feature_columns = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
                   'temperature', 'humidity', 'rain',
                   'traffic_flow_lag_1h', 'traffic_flow_lag_1d']
target_column = 'traffic_flow'

# Drop rows with NaN (if any exist)
processed_data = traffic_data.dropna(subset=feature_columns + [target_column])

# Split into features (X) and target (y)
X = processed_data[feature_columns]
y = processed_data[target_column]

# Perform an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Reshape the data for LSTM
# Assuming X_train, X_val, and X_test are feature matrices, and y_train, y_val, and y_test are target arrays
# Timesteps = 1 for single time-step prediction
X_train_lstm = np.expand_dims(X_train, axis=1)
X_val_lstm = np.expand_dims(X_val, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

print(f"Reshaped X_train shape: {X_train_lstm.shape}")
print(f"Reshaped X_val shape: {X_val_lstm.shape}")
print(f"Reshaped X_test shape: {X_test_lstm.shape}")

# Build the LSTM model
model = Sequential()

# Input Layer and First Hidden LSTM Layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))  # Dropout for regularization

# Second Hidden LSTM Layer
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Dense Layer (Fully Connected Layer)
model.add(Dense(units=16, activation='relu'))

# Output Layer
model.add(Dense(units=1))  # Regression output

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Model Summary
model.summary()

# Train the model
history = model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_lstm, y_test, verbose=1)

print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")



# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
predicted_traffic = model.predict(X_test_lstm)

# Compare predicted vs. actual traffic
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Traffic Flow', color='blue')
plt.plot(predicted_traffic, label='Predicted Traffic Flow', color='orange')
plt.title('Actual vs. Predicted Traffic Flow')
plt.xlabel('Time')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()



# Example model architecture
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1)  # Regression output
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Use Adam optimizer
    loss='mean_squared_error',           # Loss function
    metrics=['mae']                      # Metrics
)

# Display the model summary
model.summary()


