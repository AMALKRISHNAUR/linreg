import pandas as pd

# Load the dataset
data = pd.read_csv('house.csv')  # Replace with your dataset path

# Remove rows with NaN values
data = data.dropna()

# Split into X (features) and y (target)
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# Ensure there are at least 256 rows
if len(data) < 256:
    print("Error: Dataset has fewer than 256 rows after removing NaN.")
else:
    # Take the first 256 rows for training
    X_train = X.iloc[:256]
    y_train = y.iloc[:256]
    
    # Take the remaining rows for testing
    X_test = X.iloc[256:]
    y_test = y.iloc[256:]
    
    # Save the splits to CSV files
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    # Print shapes to verify
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Test data shape:", X_test.shape, y_test.shape)