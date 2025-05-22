import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# X values: number of nodes (same as original)
nodes = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]

# Y values: execution times in ms (example values, replace with your actual times)
times = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]

# Create the figure with high DPI
plt.figure(figsize=(10, 8), dpi=300)

# Create the scatter plot with same styling
plt.scatter(nodes, times, color='blue', s=150)

# Calculate the line of best fit (y = mx, forcing through origin)
# Use sum of products divided by sum of squares to get slope (no intercept)
slope = np.sum(np.array(nodes) * np.array(times)) / np.sum(np.array(nodes) ** 2)

# Generate points for the line of best fit
x_line = np.linspace(0, max(nodes), 100)  # Start from 0 for the origin
y_line = slope * x_line  # No intercept term

# Plot the line of best fit
plt.plot(x_line, y_line, 'r-', linewidth=4, alpha=0.85,
         label=f'Linear fit (slope = {slope:.2f})')

# Set both axes to log scale
plt.xscale('log')
plt.yscale('log')

# Set axis labels with font size 16
plt.xlabel('Monte Carlo Tree Search Iterations', fontsize=16)
plt.ylabel('Search Time (ms)', fontsize=16)

# Set tick font size to 16
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Format x-axis to display actual values not powers
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().xaxis.set_major_formatter(formatter)

# Add a legend in the top left
plt.legend(fontsize=16, loc='upper left', frameon=False)

# Adjust layout
plt.tight_layout()

# Save the figure with 300 DPI resolution
plt.savefig('eval_creation_code/nodes_vs_time_log_log.png', dpi=300)