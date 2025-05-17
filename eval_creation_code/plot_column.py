import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "unguided MDRNN", 
    "key + mode", 
    "tempo + swing", 
    "pitch interval", 
    "time multiple", 
    "repetition", 
    "combined heuristic"
]

# NOTE
# individual, total, improv
#values = [3.02, 2.58, 3.14, 3.90, 3.07, 3.49, 5.06]
# individual, pitch, improv
#values = [9.84, 11.44, 9.22, 16.12, 8.97, 11.60, 18.82]
# individual, time, improv
#values = [26.27, 17.50, 26.50, 19.10, 30.72, 26.19, 22.54]
# combined, total, improv
#values = [3.02, 4.40, 4.29, 3.99, 4.64, 5.00, 5.06]
# individual, total, nottingham
#values = [6.02, 7.46, 8.39, 6.77, 8.02, 11.24, 14.76]
# individual, pitch, nottingham
values = [12.16, 20.21, 12.33, 20.74, 12.18, 18.83, 22.13]
# individual, time, nottingham
#values = [51.54, 37.62, 70.40, 34.21, 65.56, 51.85, 67.55]
# combined, total, nottingham
#values = [6.02, 13.23, 14.01, 13.97, 14.61, 13.06, 14.76]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
plt.rcParams.update({'font.size': 16})

# Define colors
# NOTE
#bar_color = '#64B5CD'  # Improv bar color
bar_color =  '#8172B2'  # Nottingham bar color
unguided_line_color = '#C44E52'  # Red for unguided reference line
combined_line_color = '#55A868'  # Green for combined heuristic reference line

# Get values for horizontal lines
unguided_value = values[0]  # Value for unguided MDRNN
combined_value = values[-1]  # Value for combined heuristic

# Create bars (excluding the first and last items which will be reference lines)
bar_labels = labels[1:-1]
bar_values = values[1:-1]
bar_positions = np.arange(len(bar_labels))

# Plot bars
bars = ax.bar(bar_positions, bar_values, color=bar_color, width=0.6)

# Add horizontal reference lines across the entire plot
ax.axhline(y=unguided_value, color=unguided_line_color, linestyle='-', linewidth=4)
ax.axhline(y=combined_value, color=combined_line_color, linestyle='-', linewidth=4)

# Set labels and title
# NOTE
ax.set_ylabel('Pitch Prediction Accuracy (%)', fontsize=16)

# Set x-axis tick labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(bar_labels, fontsize=16)

# Create legend for reference lines
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=unguided_line_color, lw=4, label=f'Unguided MDRNN ({unguided_value:.2f}%)'),
    Line2D([0], [0], color=combined_line_color, lw=4, label=f'Combined Heuristic ({combined_value:.2f}%)')
]
# NOTE
#ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.96))
ax.legend(handles=legend_elements, loc='upper left')

# Adjust y-axis to include some padding above the highest value
max_value = max(values)
# NOTE
#ax.set_ylim(0, max_value * 1.03)
ax.set_ylim(0, max_value * 1.23)
#ax.set_ylim(0, max_value * 1.28)

# Add value labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    # NOTE
    # For combiened total notthingham
    #if i == 1 or i == 2:
    #    ax.text(bar.get_x() + bar.get_width()/2., height - 0.1,
    #        f'{bar_values[i]:.2f}%', ha='center', va='bottom')
    #elif i == 3:
    #    ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
    #        f'{bar_values[i]:.2f}%', ha='center', va='bottom')
    #else:

    # individual, pitch, improv
    #if i == 1:
    #    ax.text(bar.get_x() + bar.get_width()/2., height + 0.55,
    #        f'{bar_values[i]:.2f}%', ha='center', va='bottom')
    #elif i == 3:
    #    ax.text(bar.get_x() + bar.get_width()/2., height - 0.20,
    #        f'{bar_values[i]:.2f}%', ha='center', va='bottom')
    #else:

    # individual, time, nottingham
    #if i == 3:
    #    ax.text(bar.get_x() + bar.get_width()/2., height + 1.75,
    #        f'{bar_values[i]:.2f}%', ha='center', va='bottom')
    #else:
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{bar_values[i]:.2f}%', ha='center', va='bottom')

# Adjust layout to fit everything nicely
plt.tight_layout()

# Save the figure
plt.savefig('eval_creation_code/column.png', dpi=300, bbox_inches='tight')