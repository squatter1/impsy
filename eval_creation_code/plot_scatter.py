
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

# Generate sample data - replace this with your actual data
# X values: number of nodes (generated on a logarithmic scale)
nodes = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]
#nodes = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]
#accuracy = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]
# Y values: prediction accuracy (random values between 70-95%)
accuracy = [2.92, 2.95, 3.13, 3.84, 3.97, 4.19, 4.61, 4.72, 4.92, 5.26, 5.34, 5.56, 5.34]
#accuracy = [5.95, 6.14, 8.14, 9.94, 10.84, 12.18, 14.08, 14.55, 15.39, 16.14, 16.34, 16.94, 16.73]

# Define a function that approaches an asymptote
def asymptotic_func(x, a, b, c):
    """
    Function that approaches asymptote 'a': y = a - b/(x^c)
    This provides a smooth approach to the asymptote.
    
    Parameters:
    - a: asymptotic maximum (the horizontal asymptote)
    - b: scaling parameter
    - c: rate parameter (controls how quickly curve approaches asymptote)
    """
    return a - b/(np.power(x, c))

# Fit the function to the data with multiple starting points to avoid local minima
best_residual = np.inf
best_params = None

# Try several initial parameter sets to find the best fit
initial_guesses = []
# Fill initial_guesses with all combinations of a, b, c with 18 <= a <= 25 (moving up by 1), 10 <= b <= 90 (moving up by 10), 0.1 <= c <= 1.0 (moving up by 0.1)
for a in range(6, 25):
    for b in range(10, 100, 10):
        for c in np.arange(0.1, 1.1, 0.1):
            initial_guesses.append([a, b, c])

for p0 in initial_guesses:
    try:
        popt, pcov = curve_fit(
            asymptotic_func, 
            nodes, 
            accuracy,
            p0=p0,
            maxfev=10000  # Increase maximum function evaluations
        )
        # Calculate residual sum of squares to evaluate fit quality
        residual = np.sum((asymptotic_func(nodes, *popt) - accuracy)**2)
        
        if residual < best_residual:
            best_residual = residual
            best_params = popt
    except RuntimeError:
        continue

# Use the best parameters found
if best_params is not None:
    a_opt, b_opt, c_opt = best_params
    fit_successful = True
else:
    # Fallback parameters if fitting completely fails
    print("Curve fitting failed with all initial guesses. Using fallback parameters.")
    a_opt, b_opt, c_opt = 90, 30, 0.2
    fit_successful = False

# Generate smooth curve for plotting
x_smooth = np.logspace(np.log10(min(nodes)), np.log10(max(nodes)*2), 300)
y_fit = asymptotic_func(x_smooth, a_opt, b_opt, c_opt)

# Calculate R-squared to evaluate goodness of fit
y_pred = asymptotic_func(nodes, a_opt, b_opt, c_opt)
ss_total = np.sum((accuracy - np.mean(accuracy))**2)
ss_residual = np.sum((accuracy - y_pred)**2)
r_squared = 1 - (ss_residual / ss_total)

# Create the figure with high DPI
plt.figure(figsize=(10, 8), dpi=300)

# Create the scatter plot
plt.scatter(nodes, accuracy, color='blue', s=80, alpha=0.7, label='Data')

# Add the asymptotic curve fit
plt.plot(x_smooth, y_fit, 'r-', linewidth=2, label=f'Asymptotic Fit (Max ≈ {a_opt:.1f}%)')

# Set axis labels with font size 16
plt.xlabel('Number of Nodes', fontsize=16)
plt.ylabel('Prediction Accuracy (%)', fontsize=16)

# Set x-axis to log scale
plt.xscale('log')

# Set tick font size to 16
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Format x-axis to display actual values not powers
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().xaxis.set_major_formatter(formatter)

# Add a legend
plt.legend(fontsize=16)

# Add R-squared value in bottom right corner
plt.text(0.7, 0.05, f"R² = {r_squared:.3f}", 
         transform=plt.gca().transAxes, fontsize=16)

# Adjust layout
plt.tight_layout()

# Save the figure with 300 DPI resolution
plt.savefig('eval_creation_code/prediction_accuracy_vs_nodes.png', dpi=300)