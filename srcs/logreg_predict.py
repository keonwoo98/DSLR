#!/usr/bin/env python3
"""
logreg_predict.py - Predict Hogwarts houses using trained logistic regression model

Usage:
    python logreg_predict.py <dataset_test.csv> <weights.csv>

Output:
    houses.csv - Predicted houses for each student

Goal: Use trained model to predict houses for new students
- Load trained weights from weights.csv
- Apply same normalization as training
- Predict house with highest probability
"""

import sys
import math

# Reuse functions from previous scripts
from describe import read_csv, extract_numeric_columns


def load_weights(filename='weights.csv'):
    """
    Load trained weights from CSV file

    File format:
    house,bias,feature1_weight,feature2_weight,...
    Gryffindor,0.5,0.3,-0.2,...
    ...

    # Normalization parameters
    means,mean1,mean2,...
    stds,std1,std2,...

    Args:
        filename: Path to weights file

    Returns:
        tuple: (all_weights, means, stds, feature_names, houses)
            - all_weights: {house: [w₀, w₁, ...]}
            - means: [mean1, mean2, ...]
            - stds: [std1, std2, ...]
            - feature_names: [feature1, feature2, ...]
            - houses: [house1, house2, ...]
    """
    print(f"Loading weights from {filename}...")

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header (feature names)
    header = lines[0].strip().split(',')
    feature_names = header[2:]  # Skip 'house' and 'bias'

    # Parse weights for each house
    all_weights = {}
    houses = []
    line_idx = 1

    while line_idx < len(lines):
        line = lines[line_idx].strip()

        # Stop at normalization parameters section
        if line.startswith('#') or line.startswith('means'):
            break

        if line:  # Skip empty lines
            parts = line.split(',')
            house = parts[0]
            weights = [float(w) for w in parts[1:]]  # bias + feature weights

            all_weights[house] = weights
            houses.append(house)

        line_idx += 1

    # Find and parse normalization parameters
    means = None
    stds = None

    for i in range(line_idx, len(lines)):
        line = lines[i].strip()

        if line.startswith('means,'):
            means = [float(m) for m in line.split(',')[1:]]
        elif line.startswith('stds,'):
            stds = [float(s) for s in line.split(',')[1:]]

    if means is None or stds is None:
        print("Error: Could not find normalization parameters in weights file")
        sys.exit(1)

    print(f"✓ Loaded weights for {len(houses)} houses")
    print(f"✓ Features: {len(feature_names)} subjects")

    return all_weights, means, stds, feature_names, houses


def load_test_data(filename, feature_names):
    """
    Read test CSV file and return student info and features

    Args:
        filename: CSV file path
        feature_names: List of feature names (must match training)

    Returns:
        tuple: (student_ids, features)
            - student_ids: List of student indices
            - features: List of [score1, score2, ...] for each student
    """
    print(f"\nReading test file: {filename}")

    # 1. Read CSV
    data = read_csv(filename)
    header = data[0]
    rows = data[1:]

    # 2. Find Index column
    try:
        index_col = header.index('Index')
    except ValueError:
        print("Error: 'Index' column not found in test file")
        sys.exit(1)

    # 3. Extract numeric columns
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    # Exclude Index column from features
    start_idx = 1 if numeric_cols[0] == 'Index' else 0

    # 4. Prepare features for each student
    student_ids = []
    features = []

    for row_idx, row in enumerate(rows):
        student_id = row[index_col]
        student_ids.append(student_id)

        # Get feature values for this student
        student_features = []

        for feat_name in feature_names:
            # Find feature in numeric columns
            try:
                feat_idx = numeric_cols.index(feat_name, start_idx)
                value = numeric_data[feat_idx][row_idx]

                # Handle missing values: use 0 after normalization
                if value is None:
                    value = 0.0

                student_features.append(value)

            except ValueError:
                print(f"Error: Feature '{feat_name}' not found in test file")
                sys.exit(1)

        features.append(student_features)

    print(f"Loaded {len(student_ids)} students from test file")

    return student_ids, features


def normalize_features(features, means, stds):
    """
    Normalize features using training mean and std

    IMPORTANT: Use the SAME mean and std from training!

    Formula: normalized = (value - mean) / std

    Args:
        features: List of [score1, score2, ...] for each student
        means: Mean values from training
        stds: Std values from training

    Returns:
        list: Normalized features
    """
    print(f"Normalizing features using training parameters...")

    normalized = []

    for student_features in features:
        normalized_student = []

        for feat_idx, value in enumerate(student_features):
            # Apply same normalization as training
            normalized_value = (value - means[feat_idx]) / stds[feat_idx]
            normalized_student.append(normalized_value)

        normalized.append(normalized_student)

    print(f"✓ Normalization complete")

    return normalized


def sigmoid(z):
    """
    Sigmoid activation function

    Formula: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: Input value

    Returns:
        float: Value between 0 and 1
    """
    # Handle overflow
    if z < -500:
        return 0.0
    elif z > 500:
        return 1.0

    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(features, weights):
    """
    Predict probability using logistic regression

    Formula: p = σ(w₀ + w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ)

    Args:
        features: [x₁, x₂, ..., xₙ] - feature values for one student
        weights: [w₀, w₁, w₂, ..., wₙ] - model parameters

    Returns:
        float: Probability (0 to 1)
    """
    # w₀ is bias (intercept), w₁...wₙ are feature weights
    z = weights[0]  # Start with bias

    # Add weighted sum of features
    for i, feature_value in enumerate(features):
        z += weights[i + 1] * feature_value

    # Apply sigmoid
    probability = sigmoid(z)

    return probability


def predict_houses(features, all_weights, houses):
    """
    Predict house for each student

    For each student:
    1. Calculate probability for each house
    2. Choose house with highest probability

    Args:
        features: List of normalized features for all students
        all_weights: Dictionary {house: weights}
        houses: List of house names

    Returns:
        list: Predicted house for each student
    """
    print(f"\nPredicting houses for {len(features)} students...")

    predictions = []

    for student_features in features:
        # Calculate probabilities for all houses
        probabilities = {}

        for house in houses:
            weights = all_weights[house]
            prob = predict_probability(student_features, weights)
            probabilities[house] = prob

        # Choose house with highest probability
        predicted_house = max(probabilities, key=probabilities.get)
        predictions.append(predicted_house)

    print(f"✓ Predictions complete")

    return predictions


def save_predictions(student_ids, predictions, filename='houses.csv'):
    """
    Save predictions to CSV file

    Format:
    Index,Hogwarts House
    0,Gryffindor
    1,Hufflepuff
    ...

    Args:
        student_ids: List of student indices
        predictions: List of predicted houses
        filename: Output file path
    """
    print(f"\nSaving predictions to {filename}...")

    with open(filename, 'w') as f:
        # Header
        f.write('Index,Hogwarts House\n')

        # Predictions
        for student_id, house in zip(student_ids, predictions):
            f.write(f'{student_id},{house}\n')

    print(f"✓ Predictions saved to {filename}")


def main():
    # Check usage
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_test.csv> <weights.csv>")
        sys.exit(1)

    test_file = sys.argv[1]
    weights_file = sys.argv[2]

    print("=" * 80)
    print("LOGISTIC REGRESSION PREDICTION")
    print("=" * 80)

    # 1. Load trained model
    all_weights, means, stds, feature_names, houses = load_weights(weights_file)

    # 2. Load test data
    student_ids, features = load_test_data(test_file, feature_names)

    # 3. Normalize features (using training parameters)
    normalized_features = normalize_features(features, means, stds)

    # 4. Predict houses
    predictions = predict_houses(normalized_features, all_weights, houses)

    # 5. Save predictions
    save_predictions(student_ids, predictions)

    print(f"\n{'=' * 80}")
    print("PREDICTION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
