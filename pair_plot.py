#!/usr/bin/env python3
"""
pair_plot.py - Visualize pairwise relationships between features

Usage:
    python pair_plot.py <dataset.csv>

Output:
    1. pair_plot_all.png - All 13 subjects (complete overview)
    2. pair_plot_best.png - Top 6 most significant subjects (detailed answer)

Goal: Visualize relationships between multiple features at once
- Diagonal: Distribution of each subject (histogram)
- Off-diagonal: Scatter plots between pairs of subjects
- Helps identify patterns and feature relationships
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


def calculate_homogeneity_score(house_data, feature):
    """
    Calculate how different the distributions are across houses
    (Same as histogram.py)

    Lower score = more homogeneous (similar across houses)
    Higher score = better distinguisher (different across houses)
    """
    means = []

    for house in house_data.keys():
        scores = house_data[house][feature]
        if len(scores) > 0:
            mean = sum(scores) / len(scores)
            means.append(mean)

    if len(means) == 0:
        return 0.0

    # Calculate standard deviation of means
    mean_of_means = sum(means) / len(means)
    variance = sum((m - mean_of_means) ** 2 for m in means) / len(means)
    std_of_means = variance ** 0.5

    return std_of_means


def calculate_correlation(x_values, y_values):
    """
    Calculate Pearson correlation coefficient
    (Same as scatter_plot.py)
    """
    if len(x_values) == 0 or len(y_values) == 0:
        return 0.0

    n = min(len(x_values), len(y_values))
    x_values = x_values[:n]
    y_values = y_values[:n]

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_average_correlation(feature, all_features, house_data):
    """
    Calculate average absolute correlation with all other features

    Lower score = more independent
    """
    correlations = []

    for other_feature in all_features:
        if other_feature == feature:
            continue

        # Combine all houses
        all_x = []
        all_y = []

        for house in house_data.keys():
            scores_x = house_data[house][feature]
            scores_y = house_data[house][other_feature]

            min_len = min(len(scores_x), len(scores_y))
            all_x.extend(scores_x[:min_len])
            all_y.extend(scores_y[:min_len])

        corr = calculate_correlation(all_x, all_y)
        correlations.append(abs(corr))

    if len(correlations) == 0:
        return 0.0

    return sum(correlations) / len(correlations)


def select_best_features(all_features, house_data):
    """
    Automatically select best features based on:
    1. Top 3 distinguishers (high homogeneity score = good for classification)
    2. Top 3 independent (low average correlation = diverse information)

    Returns:
        list: Selected feature names (6 features)
    """
    print("\nAnalyzing all subjects...")

    # 1. Calculate homogeneity scores (for distinguishers)
    homogeneity_scores = []
    for feature in all_features:
        score = calculate_homogeneity_score(house_data, feature)
        homogeneity_scores.append((score, feature))

    # Sort by score (descending) - higher = better distinguisher
    homogeneity_scores.sort(reverse=True)

    # 2. Calculate independence scores
    independence_scores = []
    for feature in all_features:
        avg_corr = calculate_average_correlation(feature, all_features, house_data)
        independence_scores.append((avg_corr, feature))

    # Sort by score (ascending) - lower = more independent
    independence_scores.sort()

    # 3. Select top 3 from each category
    top_distinguishers = [f for _, f in homogeneity_scores[:3]]
    top_independent = [f for _, f in independence_scores[:3]]

    print("\nTop 3 Distinguishers (best for house classification):")
    for i, (score, feature) in enumerate(homogeneity_scores[:3], 1):
        print(f"  {i}. {feature} (score: {score:.2f})")

    print("\nTop 3 Independent Features (most diverse information):")
    for i, (score, feature) in enumerate(independence_scores[:3], 1):
        print(f"  {i}. {feature} (avg correlation: {score:.4f})")

    # 4. Combine (remove duplicates if any)
    selected = []
    for f in top_distinguishers + top_independent:
        if f not in selected:
            selected.append(f)

    return selected


def plot_pair_plot(selected_features, house_data, colors, output_file, title):
    """
    Generate pair plot for given features
    """
    n_features = len(selected_features)

    print(f"\nGenerating {output_file} ({n_features}x{n_features} grid)...")

    # Create figure with subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=(18, 18))

    # Handle single feature case (shouldn't happen, but just in case)
    if n_features == 1:
        axes = [[axes]]
    elif n_features == 2:
        axes = axes.reshape(2, 2)

    # Plot each cell in the grid
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i][j]

            feature_y = selected_features[i]  # Row = Y axis
            feature_x = selected_features[j]  # Column = X axis

            if i == j:
                # Diagonal: Histogram of single feature
                for house in house_data.keys():
                    scores = house_data[house][feature_x]
                    if len(scores) > 0:
                        ax.hist(scores, bins=20, alpha=0.5,
                               color=colors[house], label=house)

                ax.set_ylabel('Frequency', fontsize=8)

            else:
                # Off-diagonal: Scatter plot between two features
                for house in house_data.keys():
                    scores_x = house_data[house][feature_x]
                    scores_y = house_data[house][feature_y]

                    # Align data
                    min_len = min(len(scores_x), len(scores_y))
                    scores_x = scores_x[:min_len]
                    scores_y = scores_y[:min_len]

                    ax.scatter(scores_x, scores_y,
                              alpha=0.4, s=5,
                              color=colors[house])

            # Labels
            if i == n_features - 1:  # Bottom row
                # Truncate long names
                label = feature_x if len(feature_x) <= 15 else feature_x[:12] + "..."
                ax.set_xlabel(label, fontsize=9, fontweight='bold')
            else:
                ax.set_xticklabels([])

            if j == 0:  # Left column
                # Truncate long names
                label = feature_y if len(feature_y) <= 15 else feature_y[:12] + "..."
                ax.set_ylabel(label, fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=colors[house], markersize=10, label=house)
              for house in house_data.keys()]
    fig.legend(handles=handles, loc='upper right', fontsize=12, framealpha=0.9)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def main():
    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    print("=" * 80)
    print("PAIR PLOT ANALYSIS - Visualizing Feature Relationships")
    print("=" * 80)

    print(f"\nReading file: {filename}")
    all_features, house_data = load_data_with_houses(filename)

    print(f"Found {len(all_features)} subjects")

    # House colors
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    # 1. Generate pair_plot_all.png with ALL features
    plot_pair_plot(all_features, house_data, colors,
                   'pair_plot_all.png',
                   'Pair Plot: All Subjects')

    # 2. Select best features and generate pair_plot_best.png
    best_features = select_best_features(all_features, house_data)

    print(f"\n{'=' * 80}")
    print(f"SELECTED FEATURES FOR DETAILED ANALYSIS")
    print(f"{'=' * 80}")
    for i, feature in enumerate(best_features, 1):
        print(f"  {i}. {feature}")
    print(f"{'=' * 80}\n")

    plot_pair_plot(best_features, house_data, colors,
                   'pair_plot_best.png',
                   'Pair Plot: Most Significant Subjects')

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ pair_plot_all.png  - All {len(all_features)} subjects")
    print(f"✓ pair_plot_best.png - Top {len(best_features)} most significant subjects")
    print(f"\nHow to read:")
    print(f"  • Diagonal: Distribution of each subject")
    print(f"  • Off-diagonal: Relationships between pairs of subjects")
    print(f"  • Colors: Gryffindor (red), Hufflepuff (yellow),")
    print(f"            Ravenclaw (blue), Slytherin (green)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
