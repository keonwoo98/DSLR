#!/usr/bin/env python3
"""
scatter_plot.py - Analyze correlation between two subjects using scatter plots

Usage:
    python scatter_plot.py <dataset.csv>

Output:
    1. scatter_plot_all.png - All 78 subject pairs (overview)
    2. scatter_plot_best.png - Most correlated pair (detailed answer)

Goal: Find which two subjects are most similar
- Plot scores of subject A (x-axis) vs subject B (y-axis)
- If two subjects have similar information, students will show a linear pattern
- Most similar subjects = redundant features (can use just one)
"""

import sys
import matplotlib.pyplot as plt

# Reuse functions from previous scripts
from describe import read_csv, extract_numeric_columns


def load_data_with_houses(filename):
    """
    Read CSV file and return data classified by house

    Args:
        filename: CSV file path

    Returns:
        tuple: (feature_names, house_data)
            - feature_names: List of subject names
            - house_data: {house: {subject: [scores]}}
    """
    # 1. Read CSV
    data = read_csv(filename)
    header = data[0]
    rows = data[1:]

    # 2. Find house column index
    house_idx = header.index('Hogwarts House')

    # 3. Extract numeric columns
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    # Exclude Index column
    start_idx = 1 if numeric_cols[0] == 'Index' else 0
    feature_names = numeric_cols[start_idx:]

    # 4. Create nested dictionary: {house: {subject: [scores]}}
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    house_data = {house: {feat: [] for feat in feature_names} for house in houses}

    # 5. Classify data by house
    for row_idx, row in enumerate(rows):
        house = row[house_idx]
        if house not in houses:
            continue

        for feat_idx, feat_name in enumerate(feature_names):
            col_idx = start_idx + feat_idx
            value = numeric_data[col_idx][row_idx]
            if value is not None:
                house_data[house][feat_name].append(value)

    return feature_names, house_data


def calculate_correlation(x_values, y_values):
    """
    Calculate Pearson correlation coefficient

    Returns:
        float: Correlation coefficient (-1.0 to +1.0)
    """
    if len(x_values) == 0 or len(y_values) == 0:
        return 0.0

    n = min(len(x_values), len(y_values))
    x_values = x_values[:n]
    y_values = y_values[:n]

    # Calculate means
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    # Calculate correlation
    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))

    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def find_best_correlated_pair(feature_names, house_data):
    """
    Find the pair with highest absolute correlation

    Returns:
        tuple: (correlation, subject1, subject2)
    """
    best_corr = 0.0
    best_pair = (None, None)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            subject1 = feature_names[i]
            subject2 = feature_names[j]

            # Combine all houses for correlation calculation
            all_x = []
            all_y = []

            for house in house_data.keys():
                scores_x = house_data[house][subject1]
                scores_y = house_data[house][subject2]

                min_len = min(len(scores_x), len(scores_y))
                all_x.extend(scores_x[:min_len])
                all_y.extend(scores_y[:min_len])

            corr = calculate_correlation(all_x, all_y)

            # Check if this is the best correlation
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_pair = (subject1, subject2)

    return best_corr, best_pair[0], best_pair[1]


def plot_all_pairs(feature_names, house_data, colors):
    """
    Generate scatter_plot_all.png with all 78 pairs
    """
    print(f"Generating scatter_plot_all.png (all {len(feature_names)}C2 pairs)...")

    # Calculate all possible pairs
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            pairs.append((feature_names[i], feature_names[j]))

    # Create subplot grid (9x9 = 81 subplots for 78 pairs)
    rows, cols = 9, 9
    fig, axes = plt.subplots(rows, cols, figsize=(35, 35))
    axes = axes.flatten()

    # Plot each pair
    for idx, (subject_x, subject_y) in enumerate(pairs):
        ax = axes[idx]

        # Plot each house with different color
        for house in house_data.keys():
            scores_x = house_data[house][subject_x]
            scores_y = house_data[house][subject_y]

            # Align data: only use students who have both scores
            min_len = min(len(scores_x), len(scores_y))
            scores_x = scores_x[:min_len]
            scores_y = scores_y[:min_len]

            # Draw scatter plot
            ax.scatter(scores_x, scores_y,
                      alpha=0.4,
                      s=2,
                      color=colors[house],
                      label=house)

        # Set labels and title
        ax.set_xlabel(subject_x, fontsize=5)
        ax.set_ylabel(subject_y, fontsize=5)
        ax.set_title(f"{subject_x[:12]}... vs {subject_y[:12]}...", fontsize=6)
        ax.tick_params(labelsize=4)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save visualization
    output_file = 'scatter_plot_all.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def plot_best_pair(subject1, subject2, correlation, house_data, colors):
    """
    Generate scatter_plot_best.png with the most correlated pair
    """
    print(f"Generating scatter_plot_best.png ({subject1} vs {subject2})...")

    # Create large single plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each house with different color
    for house in house_data.keys():
        scores_x = house_data[house][subject1]
        scores_y = house_data[house][subject2]

        # Align data
        min_len = min(len(scores_x), len(scores_y))
        scores_x = scores_x[:min_len]
        scores_y = scores_y[:min_len]

        # Draw scatter plot with larger points
        ax.scatter(scores_x, scores_y,
                  alpha=0.6,
                  s=50,
                  color=colors[house],
                  label=house,
                  edgecolors='black',
                  linewidth=0.5)

    # Set labels and title
    ax.set_xlabel(subject1, fontsize=14, fontweight='bold')
    ax.set_ylabel(subject2, fontsize=14, fontweight='bold')
    ax.set_title(f"Most Similar Features: {subject1} vs {subject2}\n(Correlation: {correlation:+.4f})",
                fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)

    plt.tight_layout()

    # Save visualization
    output_file = 'scatter_plot_best.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def main():
    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    print("=" * 80)
    print("SCATTER PLOT ANALYSIS - Finding Most Similar Features")
    print("=" * 80)

    print(f"\nReading file: {filename}")
    feature_names, house_data = load_data_with_houses(filename)

    print(f"Found {len(feature_names)} subjects")
    print(f"Total {len(feature_names) * (len(feature_names) - 1) // 2} subject pairs to analyze")

    # House colors (same as histogram)
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    # Find best correlated pair
    print(f"\nAnalyzing correlations...")
    correlation, subject1, subject2 = find_best_correlated_pair(feature_names, house_data)

    print(f"\n{'=' * 80}")
    print(f"RESULT: Most Similar Features")
    print(f"{'=' * 80}")
    print(f"  Subject 1: {subject1}")
    print(f"  Subject 2: {subject2}")
    print(f"  Correlation: {correlation:+.4f}")
    print(f"{'=' * 80}\n")

    # Generate visualizations
    plot_all_pairs(feature_names, house_data, colors)
    plot_best_pair(subject1, subject2, correlation, house_data, colors)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ scatter_plot_all.png  - Overview of all subject pairs")
    print(f"✓ scatter_plot_best.png - Answer: {subject1} vs {subject2}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
