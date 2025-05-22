import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd

# X values: number of nodes (generated on a logarithmic scale)
nodes = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]
#nodes = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]
#accuracy = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]
# Y values: prediction accuracy (random values between 70-95%)
accuracy = [2.92, 2.95, 3.13, 3.84, 3.97, 4.19, 4.61, 4.72, 4.92, 5.26, 5.34, 5.56, 5.34]
#accuracy = [5.95, 6.14, 8.14, 9.94, 10.84, 12.18, 14.08, 14.55, 15.39, 16.14, 16.34, 16.94, 16.73]

# Convert to numpy arrays to ensure proper handling
x = np.array(nodes)
y = np.array(accuracy)

# Modified asymptotic function to fit the data pattern
# Function form: y = a * (1 - exp(-x/b)) + c
def asymptotic_func(x, a, b, c):
    #return a * (1 - np.exp(-x/b)) + c
    return a - b/(np.power(x, c))

# Try several initial parameter sets to find the best fit
initial_guesses = []
# Generate multiple initial guesses
for a in range(6, 25):
    for b in range(10, 100, 10):
        for c in np.arange(0.1, 1.1, 0.1):
            initial_guesses.append([a, b, c])

# Variables to store the best fit
best_params = None
best_residual = float('inf')
best_pcov = None

# Function to calculate R-squared
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# Try each initial guess and keep the best one with R-squared >= 0.95
for guess in initial_guesses:
    try:
        params, pcov = curve_fit(asymptotic_func, x, y, p0=guess, maxfev=10000)
        y_pred = asymptotic_func(x, *params)
        
        # Calculate R-squared
        r_squared = calculate_r_squared(y, y_pred)
        
        # Only consider fits with R-squared >= 0.95
        if r_squared >= 0.95:
            residual = np.sum((y - y_pred)**2)
            
            if residual < best_residual:
                best_residual = residual
                best_params = params
                best_pcov = pcov
                print(f"Found fit with R² = {r_squared:.4f}, params = {params}")
    except:
        continue

# If we found at least one good fit
if best_params is not None:
    params = best_params
    pcov = best_pcov
    a_fit, b_fit, c_fit = params
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    
    # Calculate the final R-squared for the best fit
    y_pred = asymptotic_func(x, *params)
    final_r_squared = calculate_r_squared(y, y_pred)
    
    print(f"Best fit parameters (R² = {final_r_squared:.4f}):")
    print(f"a (scale factor): {a_fit:.4f} ± {a_err:.4f}")
    print(f"b (rate parameter): {b_fit:.4f} ± {b_err:.4f}")
    print(f"c (offset): {c_fit:.4f} ± {c_err:.4f}")
    
    # Calculate the theoretical maximum value
    max_value = a_fit + c_fit
    print(f"Theoretical maximum: {max_value:.4f}")
else:
    print("Could not find a fit with R-squared >= 0.95. Try adjusting your parameter ranges.")
    # Exit the program early or use a default set
    import sys
    sys.exit()

# Calculate smooth curve for the best fit line
x_smooth = np.linspace(min(x), max(x)*1.1, 1000)
y_fit = asymptotic_func(x_smooth, a_fit, b_fit, c_fit)

# Method 1: Error propagation - Create confidence bands based on parameter uncertainties
def error_propagation(x, params, pcov):
    a, b, c = params
    # Calculate derivatives of the function with respect to each parameter
    da = 1 - np.exp(-x/b)
    db = a * x * np.exp(-x/b) / (b*b)
    dc = np.ones_like(x)
    
    # Create Jacobian matrix
    jacobian = np.vstack([da, db, dc])
    
    # Calculate the variance at each x point
    variance = np.sum((jacobian.T @ pcov) * jacobian.T, axis=1)
    
    # Return standard deviation
    return np.sqrt(variance)

y_err = error_propagation(x_smooth, params, pcov)

# Method 2: Bootstrap resampling to generate multiple "close fits"
n_bootstraps = 1000
bootstrap_params = []
bootstrap_max_values = []
bootstrap_curves = []
bootstrap_r_squared = []

for i in range(n_bootstraps):
    # Resample data with replacement
    indices = np.random.choice(len(x), size=len(x), replace=True)
    x_resampled = np.array([x[i] for i in indices])
    y_resampled = np.array([y[i] for i in indices])
    
    # Fit the model to the resampled data
    try:
        # Use the best parameters as starting point for bootstrap fits
        params_resampled, _ = curve_fit(asymptotic_func, x_resampled, y_resampled, 
                                       p0=params, maxfev=10000)
        
        # Calculate R-squared for this bootstrap fit
        y_pred_resampled = asymptotic_func(x_resampled, *params_resampled)
        r_squared = calculate_r_squared(y_resampled, y_pred_resampled)
        
        # Only keep fits with R-squared >= 0.95
        if r_squared >= 0.95:
            print(f"Bootstrap fit {i+1}: R² = {r_squared:.4f}, params = {params_resampled}")
            bootstrap_params.append(params_resampled)
            bootstrap_max_values.append(params_resampled[0] + params_resampled[2])  # a + c is the max value
            bootstrap_curves.append(asymptotic_func(x_smooth, *params_resampled))
            bootstrap_r_squared.append(r_squared)
    except:
        continue

# Convert lists to arrays for further processing
bootstrap_params = np.array(bootstrap_params)
bootstrap_max_values = np.array(bootstrap_max_values)
bootstrap_curves = np.array(bootstrap_curves)

print(f"Kept {len(bootstrap_params)} out of {n_bootstraps} bootstrap samples with R² >= 0.95")
# Check if we have enough bootstrap samples
if len(bootstrap_curves) < 10:
    print("Warning: Not enough bootstrap samples with R² >= 0.95. Confidence intervals may be unreliable.")
    if len(bootstrap_curves) == 0:
        print("No bootstrap samples met the R² >= 0.95 criterion. Cannot calculate confidence intervals.")
        import sys
        sys.exit()

# Calculate percentiles for the bootstrap curves
lower_bound = np.percentile(bootstrap_curves, 5, axis=0)
upper_bound = np.percentile(bootstrap_curves, 95, axis=0)

# Calculate the range for the maximum value
max_value_lower = np.percentile(bootstrap_max_values, 5)
max_value_upper = np.percentile(bootstrap_max_values, 95)

print(f"\nBootstrap 95% confidence interval for asymptotic maximum (from {len(bootstrap_curves)} samples with R² >= 0.95):")
print(f"{max_value_lower:.4f} to {max_value_upper:.4f}")

# Calculate residuals for prediction intervals
residuals = y - asymptotic_func(x, a_fit, b_fit, c_fit)
std_resid = np.std(residuals)

# Add prediction intervals (where future points might fall)
prediction_lower = y_fit - 1.96 * std_resid
prediction_upper = y_fit + 1.96 * std_resid

# Get a sample of close fits for visualization
sample_indices = np.random.choice(n_bootstraps, 50, replace=False)
sample_curves = bootstrap_curves[sample_indices]

# Plotting
plt.figure(figsize=(12, 8))

# Plot the original data points
plt.scatter(x, y, color='blue', alpha=0.8, s=50, label='Data points')

# Plot the confidence bands from error propagation
plt.fill_between(x_smooth, y_fit - 1.96 * y_err, y_fit + 1.96 * y_err, 
                color='green', alpha=0.2, label='Error propagation 95% CI')

# Plot the prediction interval
plt.fill_between(x_smooth, prediction_lower, prediction_upper, 
                color='blue', alpha=0.1, label='Prediction interval')

# Plot the best fit line
plt.plot(x_smooth, y_fit, color='red', linestyle='-', linewidth=2, label=f'Best fit (R² = {final_r_squared:.4f})')
max_value = a_fit + c_fit

# Add a title and labels
plt.title('Asymptotic Fit with Error Ranges', fontsize=14)
plt.xlabel('Number of Nodes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xscale('log')  # Use log scale for x-axis as the node numbers span several orders of magnitude
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Create a second figure that focuses on the maximum value uncertainty
if len(bootstrap_max_values) > 0:
    plt.figure(figsize=(10, 6))
    # Get rid of all outlier > 20
    bootstrap_max_values = [val for val in bootstrap_max_values if val < 20]
    # Plot histogram of bootstrap maximum values
    plt.hist(bootstrap_max_values, bins=min(30, len(bootstrap_max_values)//3), alpha=0.7, color='skyblue', 
             edgecolor='black', label=f'Bootstrap samples (R² ≥ 0.95, n={len(bootstrap_max_values)})')
    plt.axvline(x=max_value, color='red', linestyle='-', linewidth=2, label=f'Best estimate: {max_value:.4f}')

    plt.title('Distribution of Asymptotic Maximum Values (R² ≥ 0.95)', fontsize=14)
    plt.xlabel('Maximum Value (a + c)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary statistics table
    summary_stats = {
        'Statistic': ['Best Estimate (a + c)', 'Standard Error', '95% CI Lower', '95% CI Upper', 'Range Width'],
        'Value': [
            f"{max_value:.4f}", 
            f"{np.sqrt(a_err**2 + c_err**2):.4f}", 
            f"{max_value_lower:.4f}", 
            f"{max_value_upper:.4f}", 
            f"{max_value_upper - max_value_lower:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_stats)
    print("\nSummary statistics for asymptotic maximum:")
    print(summary_df.to_string(index=False))


plt.tight_layout()
plt.show()