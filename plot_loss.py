import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

csv_path = 'work_dirs/coco_allClasses_sam2_t_1_1/lightning_logs/version_4'

# Read the CSV data
data = pd.read_csv(f'{csv_path}/metrics.csv')

# Calculate time-weighted EMA
def calculate_ema(data, column, span=100):
    return data[column].ewm(span=span, adjust=False).mean()

data['iou_loss_ema'] = calculate_ema(data, 'iou_loss')
data['total_loss_ema'] = calculate_ema(data, 'total_loss')

# Set up the plot style
sns.set_palette("deep")

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the loss values and their EMAs
ax.plot(data['step'], data['iou_loss'], label='IOU Loss', alpha=0.3)
ax.plot(data['step'], data['total_loss'], label='Total Loss', alpha=0.3)
ax.plot(data['step'], data['iou_loss_ema'], label='IOU Loss EMA', linewidth=2)
ax.plot(data['step'], data['total_loss_ema'], label='Total Loss EMA', linewidth=2)

# Customize the plot
ax.set_title('Loss Values and EMAs over Training Steps', fontsize=16, fontweight='bold')
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Add annotations for minimum and maximum loss values
min_iou_loss = data['iou_loss'].min()
max_iou_loss = data['iou_loss'].max()
min_total_loss = data['total_loss'].min()
max_total_loss = data['total_loss'].max()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{csv_path}/loss_visualization_with_ema.png', dpi=300, bbox_inches='tight')
plt.close()

print("The loss visualization with EMA has been saved as 'loss_visualization_with_ema.png'.")
