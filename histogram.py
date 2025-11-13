#!/usr/bin/env python3
"""
histogram.py - Visualize score distribution by subject using histograms

Usage:
    python histogram.py <dataset.csv>

Output:
    1. histogram_all.png - All 13 subjects (overview)
    2. histogram_best.png - Most homogeneous subject (detailed answer)

Goal: Find which subject has most homogeneous score distribution between houses
- Plot score distribution for each subject by 4 houses
- Subject with similar distribution across houses = homogeneous feature
"""

import sys
import matplotlib.pyplot as plt

# Reuse functions from describe.py
from describe import read_csv, extract_numeric_columns, calculate_std


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


def calculate_homogeneity_score(house_data, feature):
    """
    Calculate how homogeneous (similar) the distributions are across houses

    Lower score = more homogeneous (all houses have similar distributions)

    Method: Calculate standard deviation of means across houses
    """
    means = []

    for house in house_data.keys():
        scores = house_data[house][feature]
        if len(scores) > 0:
            mean = sum(scores) / len(scores)
            means.append(mean)

    if len(means) == 0:
        return float('inf')

    # Calculate standard deviation of means
    mean_of_means = sum(means) / len(means)
    variance = sum((m - mean_of_means) ** 2 for m in means) / len(means)
    std_of_means = variance ** 0.5

    return std_of_means


def find_most_homogeneous_feature(feature_names, house_data):
    """
    Find the feature with most similar distribution across houses

    Returns:
        tuple: (feature_name, homogeneity_score)
    """
    best_feature = None
    best_score = float('inf')

    for feature in feature_names:
        score = calculate_homogeneity_score(house_data, feature)

        if score < best_score:
            best_score = score
            best_feature = feature

    return best_feature, best_score


def plot_all_histograms(feature_names, house_data, colors):
    """
    Generate histogram_all.png with all subjects
    """
    print(f"Generating histogram_all.png (all {len(feature_names)} subjects)...")

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
    print(f"✓ Saved: {output_file}")

    plt.close()


def plot_best_histogram(feature, homogeneity_score, house_data, colors):
    """
    Generate histogram_best.png with the most homogeneous subject
    """
    print(f"Generating histogram_best.png ({feature})...")

    # Create large single plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histogram for each house
    for house in house_data.keys():
        scores = house_data[house][feature]
        if len(scores) > 0:
            ax.hist(scores, bins=30, alpha=0.6, label=house,
                   color=colors[house], edgecolor='black', linewidth=1.2)

    # Set labels and title
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Students)', fontsize=14, fontweight='bold')
    ax.set_title(f"Most Homogeneous Feature: {feature}\n(Homogeneity Score: {homogeneity_score:.2f})",
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save
    output_file = 'histogram_best.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def main():
    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    print("=" * 80)
    print("HISTOGRAM ANALYSIS - Finding Most Homogeneous Feature")
    print("=" * 80)

    print(f"\nReading file: {filename}")
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

    # Find most homogeneous feature
    print(f"\nAnalyzing homogeneity across houses...")
    best_feature, homogeneity_score = find_most_homogeneous_feature(feature_names, house_data)

    print(f"\n{'=' * 80}")
    print(f"RESULT: Most Homogeneous Feature")
    print(f"{'=' * 80}")
    print(f"  Subject: {best_feature}")
    print(f"  Homogeneity Score: {homogeneity_score:.2f} (lower = more similar)")
    print(f"{'=' * 80}\n")

    # Generate visualizations
    plot_all_histograms(feature_names, house_data, colors)
    plot_best_histogram(best_feature, homogeneity_score, house_data, colors)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ histogram_all.png  - Overview of all subjects")
    print(f"✓ histogram_best.png - Answer: {best_feature}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
