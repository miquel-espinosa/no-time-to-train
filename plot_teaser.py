import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set global font
plt.rcParams['font.family'] = 'Noto Sans'
plt.rcParams['font.size'] = 14

# Data
datasets = ["ArTaxOr", "Clipart1k", "DIOR", "DeepFish", "NEU-DET", "UODD"]
past_methods = {
    "Meta-RCNN":      [2.8,  None, 7.8,  None,   None,   3.6],
    "TFA w/cos":      [3.1,  None, 8.0,  None,   None,   4.4],
    "FSCE":           [3.7,  None, 8.6,  None,   None,   3.9],
    "DeFRCN":         [3.6,  None, 9.3,  None,   None,   4.5],
    "Distill-cdfsod": [5.1,  7.6,  10.5, np.nan, np.nan, 5.9],
    "ViTDeT-FT":      [5.9,  6.1,  12.9, 0.9,    2.4,    4.0],
    "Detic":          [0.6,  11.4, 0.1,  0.9,    0.0,    0.0],
    "Detic-FT":       [3.2,  15.1, 4.1,  9.0,    3.8,    4.2],
    "DE-ViT":         [0.4,  0.5,  2.7,  0.4,    0.4,    1.5],
    "DE-ViT-FT":      [10.5, 13.0, 14.7, 19.3,   0.6,    2.4],
}
cd_vito = [21.0, 17.7, 17.8, 20.3, 3.6, 3.1]
ours =    [28.2, 18.9, 14.7, 30.2, 5.5, 10.0]

# Create figure
plt.figure(figsize=(13, 11))
ax = plt.gca()

# Title and labels with improved styling
plt.title("1-Shot Performance on CD-FSOD Benchmark", 
          fontsize=24, 
          fontweight="bold", 
          pad=20)

plt.ylabel("mAP (%)", 
          fontsize=18, 
          fontweight='bold', 
          labelpad=15)

# Customize grid
plt.grid(True, linestyle="--", alpha=0.3, color='gray')

# Plot past methods with a softer look
for method, scores in past_methods.items():
    plt.plot(
        range(len(datasets)),
        scores,
        marker="o",
        color="#D3D3D3",
        alpha=0.5,
        linewidth=3,
        markersize=10,
        label="Past Methods" if method == "Meta-RCNN" else "",
    )

# Plot CD-ViTO with an attractive color
plt.plot(range(len(datasets)), cd_vito, 
         linestyle="-", 
         marker="o", 
         color="#1e81b0",
         label="CD-ViTO (With Training)",
         linewidth=4,
         markersize=12)

# Plot Ours with a vibrant color
plt.plot(range(len(datasets)), ours, 
         linestyle="-", 
         color="#e28743",
         label="Ours (Training Free)",
         linewidth=4,
         marker="*",
         markersize=20,
         markeredgecolor='black',
         markeredgewidth=2,
         )

# Customize x-ticks and y-ticks with consistent font
plt.xticks(range(len(datasets)), datasets, fontsize=16, rotation=15, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Calculate improvements against second highest score
for i, our_score in enumerate(ours):
    # Get all scores for this dataset
    all_scores = []
    for method_scores in past_methods.values():
        if method_scores[i] is not None and not np.isnan(method_scores[i]):
            all_scores.append(method_scores[i])
    all_scores.append(cd_vito[i])
    
    if all_scores:  # If we have any valid scores to compare against
        second_highest = sorted(all_scores, reverse=True)[0]  # Get the highest score among others
        if second_highest > 0:  # Avoid division by zero
            percentage_diff = ((our_score - second_highest) / second_highest) * 100
            
            # Draw arrow
            arrow_props = dict(
                arrowstyle='-|>',
                color='black',
                linewidth=2,
                shrinkA=5,
                shrinkB=5
            )
            
            # Calculate arrow positions
            if percentage_diff > 0:
                y_start = second_highest
                y_end = our_score
                text_y = (y_start + y_end) / 2  # Text in middle of arrow
            else:
                y_start = second_highest
                y_end = our_score
                text_y = (y_start + y_end) / 2
            
            # Draw the arrow
            plt.annotate('', 
                        xy=(i, y_end), 
                        xytext=(i, y_start),
                        arrowprops=arrow_props)
            
            # Add percentage text with consistent font
            sign = "+" if percentage_diff > 0 else ""
            plt.text(
                i + 0.08,  # Offset text to the right of the arrow
                text_y,
                f"{sign}{percentage_diff:.1f}%",
                fontsize=14,
                color='black',
                fontweight='bold',
                ha='left',
                va='center',
                bbox=dict(facecolor='white',
                         edgecolor='none', 
                         alpha=0.7,
                         pad=0.15,
                         boxstyle='round')
            )

# Enhanced legend with consistent font
plt.legend(fontsize=16, 
          loc="upper right", 
          frameon=True, 
          facecolor='white',
          prop={'weight': 'bold'}
          )

# Final adjustments
plt.tight_layout()

# Save with high quality
plt.savefig("plots/1-shot-cdfsod.png", 
            dpi=300, 
            bbox_inches='tight')
