import numpy as np
import pandas as pd

# Load the CSV file into a Pandas DataFrame
csv_file_path = "D:\\__py_debug_temp_var_1864393704.csv"  # Replace with your file path
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File not found at path: {csv_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# Convert DataFrame to a NumPy array
data_array = df.to_numpy()

# Define the fill value
fill_value = -9999

# Function to count non-fill values in a row
def count_non_fill_values(row, fill_value):
    return np.sum(row != fill_value)

# Calculate non-fill counts for each row
non_fill_counts = np.apply_along_axis(count_non_fill_values, axis=1, arr=data_array, fill_value=fill_value)

# Identify rows where *all* values are fill_value (i.e., all -9999)
all_fill_rows = non_fill_counts == 0

# Filter out the all-fill rows
filtered_non_fill_counts = non_fill_counts[~all_fill_rows]

# Calculate the maximum, minimum, and average of the filtered counts
if filtered_non_fill_counts.size > 0:  # Ensure there are rows left after filtering
    max_non_fill = np.max(filtered_non_fill_counts)
    min_non_fill = np.min(filtered_non_fill_counts)
    average_non_fill = np.mean(filtered_non_fill_counts)
    madian =np.median(filtered_non_fill_counts)

    print("Number of non-fill values in each row (excluding all -9999 rows):")
    print(filtered_non_fill_counts)
    print(f"\nMaximum non-fill count: {max_non_fill}")
    print(f"Minimum non-fill count: {min_non_fill}")
    print(f"Average non-fill count: {average_non_fill:.2f}")
    print(madian)
    # Determine the maximum possible number of non-fill values (e.g., 365 for daily data)
    max_possible_days = 365  # Assuming daily data for a year

    # Print a message about the maximum possible days
    print(f"Maximum possible number of non-fill values (e.g., days in a year): {max_possible_days}")

else:
    print("All rows contain only the fill value.  No statistics can be calculated.")
