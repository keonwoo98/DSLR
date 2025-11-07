#!/usr/bin/env python3
"""
describe.py - Display statistical information of the dataset

Similar to pandas' describe() function:
- Count
- Mean
- Std (Standard Deviation)
- Min (Minimum)
- 25% (First Quartile)
- 50% (Median)
- 75% (Third Quartile)
- Max (Maximum)
"""

import sys


# ==================== Statistical Functions ====================

def count_values(column):
    """
    Count non-None values

    Args:
        column: List of numbers (may contain None)

    Returns:
        int: Number of valid values
    """
    count = 0
    for val in column:
        if val is not None:
            count += 1
    return count


def calculate_mean(column):
    """
    Calculate mean (average)
    Formula: (sum of all values) / (count of values)

    Args:
        column: List of numbers

    Returns:
        float: Mean value (None if empty)
    """
    # Extract non-None values only
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    # Calculate sum
    total = 0
    for val in valid_values:
        total += val

    # Mean = sum / count
    mean = total / len(valid_values)
    return mean


def calculate_min(column):
    """
    Find minimum value

    Args:
        column: List of numbers

    Returns:
        float: Minimum value
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    min_val = valid_values[0]
    for val in valid_values:
        if val < min_val:
            min_val = val

    return min_val


def calculate_max(column):
    """
    Find maximum value

    Args:
        column: List of numbers

    Returns:
        float: Maximum value
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    max_val = valid_values[0]
    for val in valid_values:
        if val > max_val:
            max_val = val

    return max_val


def calculate_std(column):
    """
    Calculate standard deviation

    Formula:
    1. Square the difference from mean → (x - mean)²
    2. Average of squared differences → variance
    3. Square root of variance → std

    Args:
        column: List of numbers

    Returns:
        float: Standard deviation
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) < 2:
        return None

    # Step 1: Calculate mean
    mean = calculate_mean(column)

    # Step 2: Square the differences from mean
    squared_diffs = []
    for val in valid_values:
        diff = val - mean
        squared_diff = diff * diff  # diff ** 2
        squared_diffs.append(squared_diff)

    # Step 3: Calculate variance (average of squared differences)
    # Using sample standard deviation (divide by n-1, same as pandas)
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)

    # Step 4: Square root (standard deviation)
    std = variance ** 0.5  # sqrt(variance)

    return std


def calculate_percentile(column, percentile):
    """
    Calculate percentile (e.g., 25%, 50%, 75%)

    Percentile: Value at a specific position when data is sorted
    Example: 25% = value at the lower 25% position

    Args:
        column: List of numbers
        percentile: Number between 0~100 (e.g., 25, 50, 75)

    Returns:
        float: Percentile value
    """
    valid_values = [val for val in column if val is not None]

    if len(valid_values) == 0:
        return None

    # Step 1: Sort in ascending order
    sorted_values = sorted(valid_values)

    # Step 2: Calculate position
    # Example: 25% of 100 data points = 25th position
    index = (percentile / 100) * (len(sorted_values) - 1)

    # Step 3: Interpolate if not an integer
    if index == int(index):
        return sorted_values[int(index)]
    else:
        # Example: 25.5th position → average of 25th and 26th
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        weight = index - lower_idx

        return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight


def print_stats_table(feature_names, stats):
    """
    Print statistics in table format like pandas.describe()

    Args:
        feature_names: List of column names
        stats: Statistics dictionary
    """
    print("\n" + "=" * 150)

    # Print header (first column is for stat names)
    header = f"{'':15}"  # Empty first column
    for name in feature_names:
        # Shorten column name if too long
        short_name = name[:12] + '...' if len(name) > 15 else name
        header += f"{short_name:>15}"
    print(header)
    print("-" * 150)

    # Print each statistics row
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    for stat_name in stat_names:
        row = f"{stat_name:15}"
        for value in stats[stat_name]:
            if value is None:
                row += f"{'NaN':>15}"
            elif stat_name == 'Count':
                # Count as integer
                row += f"{int(value):>15}"
            else:
                # Others with 2 decimal places
                row += f"{value:>15.2f}"
        print(row)

    print("=" * 150)


def read_csv(filename):
    """
    Read CSV file and return as 2D list

    Args:
        filename: CSV file path

    Returns:
        list: [header, data_row1, data_row2, ...]
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Split each line by comma
        data = []
        for line in lines:
            # Remove newline and split by comma
            row = line.strip().split(',')
            data.append(row)

        return data

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read file: {e}")
        sys.exit(1)


def is_numeric_column(values):
    """
    Check if column contains numeric data

    Args:
        values: List of column values

    Returns:
        bool: True if numeric column
    """
    numeric_count = 0
    total_count = 0

    for val in values:
        if val == '' or val == 'nan':  # Skip empty or nan values
            continue
        total_count += 1
        try:
            float(val)  # Try to convert to number
            numeric_count += 1
        except ValueError:
            pass

    # Consider as numeric if more than 50% are numbers
    if total_count == 0:
        return False
    return (numeric_count / total_count) > 0.5


def extract_numeric_columns(header, rows):
    """
    Extract only columns with numeric data

    Args:
        header: List of column names
        rows: Data rows

    Returns:
        tuple: (list of numeric column names, 2D list of numeric data)
    """
    numeric_cols = []
    numeric_indices = []

    # Check each column for numeric data
    for col_idx in range(len(header)):
        # Extract all values from this column
        column_values = [row[col_idx] for row in rows]

        if is_numeric_column(column_values):
            numeric_cols.append(header[col_idx])
            numeric_indices.append(col_idx)

    # Extract numeric data only
    numeric_data = []
    for col_idx in numeric_indices:
        column = []
        for row in rows:
            val = row[col_idx]
            # Treat empty or empty string as None
            if val == '' or val == 'nan':
                column.append(None)
            else:
                try:
                    column.append(float(val))
                except ValueError:
                    column.append(None)
        numeric_data.append(column)

    return numeric_cols, numeric_data


def main():
    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    # Step 1: Read file
    print(f"Reading file: {filename}")
    data = read_csv(filename)

    # Separate header and data
    header = data[0]
    rows = data[1:]

    print(f"Total {len(rows)} rows, {len(header)} columns")

    # Step 2: Extract numeric columns only
    print(f"\nExtracting numeric columns...")
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    print(f"Found {len(numeric_cols)} numeric columns:")
    for col_name in numeric_cols:
        print(f"   - {col_name}")

    # Step 3: Calculate statistics for each column
    print(f"\nCalculating statistics...")

    # Exclude Index column (meaningless number)
    start_idx = 1 if numeric_cols[0] == 'Index' else 0

    # Store statistics for all columns
    stats = {
        'Count': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'Max': []
    }

    # Calculate statistics for each column
    feature_names = []
    for idx in range(start_idx, len(numeric_cols)):
        col = numeric_data[idx]
        col_name = numeric_cols[idx]
        feature_names.append(col_name)

        stats['Count'].append(count_values(col))
        stats['Mean'].append(calculate_mean(col))
        stats['Std'].append(calculate_std(col))
        stats['Min'].append(calculate_min(col))
        stats['25%'].append(calculate_percentile(col, 25))
        stats['50%'].append(calculate_percentile(col, 50))
        stats['75%'].append(calculate_percentile(col, 75))
        stats['Max'].append(calculate_max(col))

    # Step 4: Print results in table format
    print_stats_table(feature_names, stats)


if __name__ == "__main__":
    main()
