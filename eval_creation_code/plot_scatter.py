import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# Generate sample data - replace this with your actual data
# X values: number of nodes (generated on a logarithmic scale)
nodes = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000] 
# Y values: prediction accuracy (random values between 70-95%)
#accuracy = [2.92, 2.95, 3.13, 3.84, 3.97, 4.19, 4.61, 4.72, 4.92, 5.26, 5.34, 5.56, 5.34]
accuracy = [5.95, 6.14, 8.14, 9.94, 10.84, 12.18, 14.08, 14.55, 15.39, 16.14, 16.34, 16.94, 16.73]

# Calculate line of best fit (using log of x values)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(nodes), accuracy)
line_fit = slope * np.log10(nodes) + intercept

# Create the figure with high DPI
plt.figure(figsize=(10, 8), dpi=300)

# Create the scatter plot
plt.scatter(nodes, accuracy, color='blue', s=80, alpha=0.7)

# Add the line of best fit
plt.plot(nodes, line_fit, 'r-', linewidth=2)

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

# Adjust layout
plt.tight_layout()

# Save the figure with 300 DPI resolution
plt.savefig('eval_creation_code/prediction_accuracy_vs_nodes.png', dpi=300)