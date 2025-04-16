import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data(x_file, y_file, w_file):
    # Read X_train
    X = pd.read_csv(x_file)
    print(f"X_train shape: {X.shape}")
    print("Sample X_train:\n", X.head(2))
    
    # Read y_train
    y = pd.read_csv(y_file)
    print(f"y_train shape: {y.shape}")
    print("Sample y_train:\n", y.head(2))
    
    # Read weights
    w = pd.read_csv(w_file)
    print(f"Weights shape: {w.shape}")
    print("Weights:\n", w)
    
    # Convert to numpy
    X = X.values
    y = y.values.flatten()
    beta = w['value'].values
    
    # Add bias column
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    print(f"X_train with bias shape: {X_bias.shape}")
    
    return X_bias, y, beta

# Compute predictions and metrics
def evaluate(X, y, beta):
    # Predict
    y_pred = X @ beta
    print("Sample predictions:", y_pred[:5])
    
    # RMSE
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    # R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    return y_pred, rmse, r2

# Plot results
def plot_results(y_true, y_pred):
    # Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual medv')
    plt.ylabel('Predicted medv')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('pred_vs_actual.png')
    plt.close()
    print("Saved pred_vs_actual.png")
    
    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, c='purple', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual medv')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig('residuals.png')
    plt.close()
    print("Saved residuals.png")

# Save metrics
def save_metrics(rmse, r2, y_pred, y_true):
    with open('train_results.csv', 'w') as f:
        f.write('metric,value\n')
        f.write(f'RMSE,{rmse:.6f}\n')
        f.write(f'R2,{r2:.6f}\n')
    print("Saved train_results.csv")
    
    # Save predictions
    pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    }).to_csv('train_predictions.csv', index=False)
    print("Saved train_predictions.csv")

def main():
    # File paths
    x_file = 'X_train.csv'
    y_file = 'y_train.csv'
    w_file = 'weights.csv'
    
    # Load data
    X, y, beta = load_data(x_file, y_file, w_file)
    
    # Evaluate
    y_pred, rmse, r2 = evaluate(X, y, beta)
    
    # Print results
    print("\nSample Predictions vs Actual (first 5):")
    print("i\tPredicted\tActual")
    for i in range(5):
        print(f"{i}\t{y_pred[i]:.4f}\t\t{y[i]:.4f}")
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training R²: {r2:.4f}")
    
    # Plot
    plot_results(y, y_pred)
    
    # Save
    save_metrics(rmse, r2, y_pred, y)

if __name__ == "__main__":
    main()