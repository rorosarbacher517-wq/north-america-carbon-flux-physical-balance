import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde


X_variables = [...]
Y_variables = [...]
ax_subtitle = [...]
model_subtitle = [...]

# Choose the index of the pair you want to plot (e.g. for the first pair)
i = 0  # Index for the specific x and y variables

x_variable = X_variables[i]
y_variable = Y_variables[i]

x = globals()[x_variable]
y = globals()[y_variable]

# Calculate kernel density estimate
xy = np.vstack([x, y]).astype(float)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# Normalize z for color mapping
z = (z - np.min(z)) / (np.max(z) - np.min(z))

# Set up the figure for a single subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Set the x and y limits based on overall data or a specific range
# You may need to define these values based on your specific dataset
overall_x_min = min(np.min(x), np.min(y))
overall_x_max = max(np.max(x), np.max(y))
overall_y_min = overall_x_min
overall_y_max = overall_x_max

ax.set_xlim([overall_x_min, overall_x_max])
ax.set_ylim([overall_y_min, overall_y_max])
ax.plot([overall_x_min, overall_x_max], [overall_y_min, overall_y_max], color='#535CA8', linestyle='--', linewidth=1)

# Scatter plot
scatter = ax.scatter(x, y, c=z, cmap='Spectral_r', s=20)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(x, y))
r2 = r2_score(x, y)
n = len(x)
bias = np.mean(x - y)
cc = np.corrcoef(x, y)[0, 1]
mean_value = np.mean(abs(x))
rrmse = rmse / mean_value

# Add text annotations
ax.text(0.05, 0.95, ax_subtitle[i], verticalalignment='top', transform=ax.transAxes,
        color='black', fontname='Times New Roman')
text = "RMSE = {:.2f}\nCC = {:.2f}\nBias = {:.2f}\nn = {}".format(rmse, cc, bias, n)
ax.text(0.6, 0.3, text, verticalalignment='top', transform=ax.transAxes,
        color='red', fontweight='bold', fontname='Times New Roman')

# Set labels
ax.set_title(model_subtitle[i], fontname='Times New Roman', pad=10)
ax.set_xlabel('Observation of NEE (g C m$^{-2}$ d$^{-1}$)', fontsize=12, fontname='Times New Roman')
ax.set_ylabel('Prediction of NEE (g C m$^{-2}$ d$^{-1}$)', fontsize=12, fontname='Times New Roman')

# Add color bar
position = fig.add_axes([0.92, 0.15, 0.015, .75])  # position [left, bottom, width, height]
cb = fig.colorbar(scatter, cax=position)
colorbarfontdict = {"size": 12, "color": "k", 'family': 'Times New Roman'}
cb.ax.set_title('Density', fontdict=colorbarfontdict, pad=10)
cb.ax.tick_params(labelsize=12, direction='in')

# Adjust overall layout
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
plt.show()

# Save the figure if needed
results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
plt.savefig(results_path + 'NEE_single_comparison_model' + '_scenario 2' + '.png', dpi=300)
