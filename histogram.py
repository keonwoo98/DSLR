#!/usr/bin/env python3
"""
histogram.py - Visualize score distribution by subject using histograms

Goal: Find which subject best distinguishes between houses
- Plot score distribution for each subject by 4 houses
- Subject with large distribution difference between houses = good feature
"""

import sys
import matplotlib.pyplot as plt

# Reuse functions from describe.py
from describe import read_csv, extract_numeric_columns


def load_data_with_houses(filename):
    """
    Read CSV file and return data with house information

    Args:
        filename: CSV file path

    Returns:
        tuple: (feature_names, house_data)
            - feature_names: List of subject names
            - house_data: {house_name: {subject_name: [scores]}}
    """
    # 1. Read CSV
    data = read_csv(filename)
    header = data[0]
    rows = data[1:]

    # 2. Find Hogwarts House column
    house_idx = header.index('Hogwarts House')

    # 3. Extract numeric columns
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    # Exclude Index column
    start_idx = 1 if numeric_cols[0] == 'Index' else 0
    feature_names = numeric_cols[start_idx:]

    # 4. Classify data by house
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    house_data = {house: {feat: [] for feat in feature_names} for house in houses}

    # Classify each student's data by house
    for row_idx, row in enumerate(rows):
        house = row[house_idx]
        if house not in houses:
            continue

        # Add scores for each subject
        for feat_idx, feat_name in enumerate(feature_names):
            col_idx = start_idx + feat_idx
            value = numeric_data[col_idx][row_idx]

            if value is not None:
                house_data[house][feat_name].append(value)

    return feature_names, house_data


def main():
    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"Reading file: {filename}")
    feature_names, house_data = load_data_with_houses(filename)

    print(f"Found {len(feature_names)} subjects")
    print(f"Houses: {list(house_data.keys())}")

    # Colors for 4 houses
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    print(f"\nGenerating histograms for all subjects...")

    # Subplot layout: 4 rows x 4 columns (enough for 13 subjects)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()  # Convert 2D array to 1D

    # Draw histogram for each subject
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]

        # Histogram for each house
        for house in house_data.keys():
            scores = house_data[house][feature]
            if len(scores) > 0:  # Only if data exists
                ax.hist(scores, bins=20, alpha=0.5, label=house, color=colors[house])

        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(feature_names), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save
    output_file = 'histogram_all.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

    print(f"\nTip: Open {output_file} to see which subject has the largest difference between houses!")


if __name__ == "__main__":
    main()
