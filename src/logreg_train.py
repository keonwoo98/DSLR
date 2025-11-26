#!/usr/bin/env python3
"""
logreg_train.py - Train logistic regression model for house classification

Usage:
    python logreg_train.py <dataset.csv> [--method METHOD] [--batch-size N]

Basic (Mandatory):
    python logreg_train.py <dataset.csv>
    # Uses SGD (Stochastic Gradient Descent) by default

BONUS (Optional - for extra credit):
    --method sgd         Stochastic Gradient Descent (default)
    --method batch       Batch Gradient Descent (BONUS 3)
    --method minibatch   Mini-batch Gradient Descent (BONUS 3)
    --batch-size N       Batch size for mini-batch (default: 32)

Output:
    weights.csv - Trained model weights

Goal: Train a logistic regression classifier to predict Hogwarts house
- Uses One-vs-All strategy for multi-class classification
- Implements gradient descent for optimization (SGD mandatory, others bonus)
- No external ML libraries (numpy, scikit-learn, etc.)
"""

import sys
import math


# ==================== Utility Functions (from describe.py) ====================

def read_csv(filename):
    """Read CSV file and return as 2D list"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
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
    """Check if column contains numeric data"""
    numeric_count = 0
    total_count = 0
    for val in values:
        if val == '' or val == 'nan':
            continue
        total_count += 1
        try:
            float(val)
            numeric_count += 1
        except ValueError:
            pass
    if total_count == 0:
        return False
    return (numeric_count / total_count) > 0.5


def extract_numeric_columns(header, rows):
    """Extract only columns with numeric data"""
    numeric_cols = []
    numeric_indices = []

    for col_idx in range(len(header)):
        column_values = [row[col_idx] for row in rows]
        if is_numeric_column(column_values):
            numeric_cols.append(header[col_idx])
            numeric_indices.append(col_idx)

    numeric_data = []
    for col_idx in numeric_indices:
        column = []
        for row in rows:
            val = row[col_idx]
            if val == '' or val == 'nan':
                column.append(None)
            else:
                try:
                    column.append(float(val))
                except ValueError:
                    column.append(None)
        numeric_data.append(column)

    return numeric_cols, numeric_data


# ==================== Training Functions ====================

def load_training_data(filename):
    """
    Read CSV file and return features and labels

    Args:
        filename: CSV file path

    Returns:
        tuple: (features, labels, feature_names, houses)
            - features: List of [score1, score2, ...] for each student
            - labels: List of house names for each student
            - feature_names: List of subject names
            - houses: List of unique house names
    """
    print(f"Reading file: {filename}")

    # 1. Read CSV
    data = read_csv(filename)
    header = data[0]
    rows = data[1:]

    # 2. Find house column
    house_idx = header.index('Hogwarts House')

    # 3. Extract numeric columns (features)
    numeric_cols, numeric_data = extract_numeric_columns(header, rows)

    # Exclude Index column
    start_idx = 1 if numeric_cols[0] == 'Index' else 0
    feature_names = numeric_cols[start_idx:]

    # 4. Prepare features and labels
    features = []
    labels = []

    for row_idx, row in enumerate(rows):
        house = row[house_idx]

        # Skip if no house label
        if not house or house.strip() == '':
            continue

        # Get feature values for this student
        student_features = []
        has_missing = False

        for feat_idx in range(len(feature_names)):
            col_idx = start_idx + feat_idx
            value = numeric_data[col_idx][row_idx]

            if value is None:
                has_missing = True
                break

            student_features.append(value)

        # Skip students with missing data (for now)
        if has_missing:
            continue

        features.append(student_features)
        labels.append(house)

    # 5. Get unique houses
    houses = sorted(list(set(labels)))

    print(f"Loaded {len(features)} students")
    print(f"Features: {len(feature_names)} subjects")
    print(f"Houses: {houses}")

    return features, labels, feature_names, houses


def normalize_features(features):
    """
    Normalize features to have mean=0 and std=1 (standardization)

    Why? Different subjects have different score ranges.
    Example: Arithmancy (0-100K), Astronomy (-1000 to +1000)

    Formula: normalized = (value - mean) / std

    Args:
        features: List of [score1, score2, ...] for each student

    Returns:
        tuple: (normalized_features, means, stds)
    """
    n_students = len(features)
    n_features = len(features[0])

    print(f"\nNormalizing {n_features} features...")

    # Calculate mean and std for each feature
    means = []
    stds = []

    for feat_idx in range(n_features):
        # Collect all values for this feature
        values = [features[i][feat_idx] for i in range(n_students)]

        # Calculate mean
        mean = sum(values) / len(values)

        # Calculate std
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        # Prevent division by zero
        if std == 0:
            std = 1.0

        means.append(mean)
        stds.append(std)

    # Normalize all features
    normalized = []
    for student_features in features:
        normalized_student = []
        for feat_idx, value in enumerate(student_features):
            normalized_value = (value - means[feat_idx]) / stds[feat_idx]
            normalized_student.append(normalized_value)
        normalized.append(normalized_student)

    print(f"✓ Normalization complete")

    return normalized, means, stds


def sigmoid(z):
    """
    Sigmoid activation function

    Formula: σ(z) = 1 / (1 + e^(-z))

    Maps any real number to range (0, 1)

    Args:
        z: Input value (can be very large or very small)

    Returns:
        float: Value between 0 and 1
    """
    # Handle overflow: if z is very negative, e^(-z) becomes huge
    # Limit z to prevent overflow
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


def train_one_vs_all_sgd(features, labels, target_house, learning_rate=0.01, epochs=1000):
    """
    Train using Stochastic Gradient Descent (SGD)
    Updates weights after EACH sample

    Pros: Fast, can escape local minima
    Cons: High variance, noisy convergence

    Args:
        features: List of [x₁, x₂, ...] for each student (normalized)
        labels: List of house names for each student
        target_house: House name to train for (e.g., 'Gryffindor')
        learning_rate: Step size for gradient descent (default: 0.01)
        epochs: Number of training iterations (default: 1000)

    Returns:
        list: Trained weights [w₀, w₁, w₂, ..., wₙ]
    """
    n_students = len(features)
    n_features = len(features[0])

    # Create binary labels: 1 if target_house, 0 otherwise
    binary_labels = [1 if label == target_house else 0 for label in labels]

    # Initialize weights (bias + n_features)
    weights = [0.0] * (n_features + 1)

    print(f"Training {target_house} vs Others (SGD)...")

    # Stochastic Gradient Descent
    for epoch in range(epochs):
        # Process each student individually
        for student_idx in range(n_students):
            student_features = features[student_idx]
            actual_label = binary_labels[student_idx]

            # 1. Predict probability
            predicted_prob = predict_probability(student_features, weights)

            # 2. Calculate error
            error = predicted_prob - actual_label

            # 3. Update weights immediately (SGD characteristic)
            weights[0] -= learning_rate * error

            for feat_idx, feature_value in enumerate(student_features):
                weights[feat_idx + 1] -= learning_rate * error * feature_value

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}")

    print(f"✓ Training complete for {target_house}")

    return weights


def train_one_vs_all_batch(features, labels, target_house, learning_rate=0.01, epochs=1000):
    """
    BONUS 3: Train using Batch Gradient Descent
    Updates weights after processing ALL samples

    Pros: Stable convergence, smooth path
    Cons: Slow, can get stuck in local minima

    Args:
        features: List of [x₁, x₂, ...] for each student (normalized)
        labels: List of house names for each student
        target_house: House name to train for
        learning_rate: Step size for gradient descent
        epochs: Number of training iterations

    Returns:
        list: Trained weights [w₀, w₁, w₂, ..., wₙ]
    """
    n_students = len(features)
    n_features = len(features[0])

    # Create binary labels
    binary_labels = [1 if label == target_house else 0 for label in labels]

    # Initialize weights
    weights = [0.0] * (n_features + 1)

    print(f"Training {target_house} vs Others (Batch GD)...")

    # Batch Gradient Descent
    for epoch in range(epochs):
        # Accumulate gradients for ALL samples
        gradient_bias = 0.0
        gradient_features = [0.0] * n_features

        # Process all students and accumulate gradients
        for student_idx in range(n_students):
            student_features = features[student_idx]
            actual_label = binary_labels[student_idx]

            # 1. Predict probability
            predicted_prob = predict_probability(student_features, weights)

            # 2. Calculate error
            error = predicted_prob - actual_label

            # 3. Accumulate gradients (not updating yet!)
            gradient_bias += error
            for feat_idx, feature_value in enumerate(student_features):
                gradient_features[feat_idx] += error * feature_value

        # 4. Update weights using averaged gradients (Batch GD characteristic)
        weights[0] -= learning_rate * (gradient_bias / n_students)
        for feat_idx in range(n_features):
            weights[feat_idx + 1] -= learning_rate * (gradient_features[feat_idx] / n_students)

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}")

    print(f"✓ Training complete for {target_house}")

    return weights


def train_one_vs_all_minibatch(features, labels, target_house, learning_rate=0.01, epochs=1000, batch_size=32):
    """
    BONUS 3: Train using Mini-batch Gradient Descent
    Updates weights after processing a SMALL BATCH of samples

    Pros: Balance between speed and stability, best of both worlds
    Cons: Need to tune batch_size

    Args:
        features: List of [x₁, x₂, ...] for each student (normalized)
        labels: List of house names for each student
        target_house: House name to train for
        learning_rate: Step size for gradient descent
        epochs: Number of training iterations
        batch_size: Number of samples per mini-batch (default: 32)

    Returns:
        list: Trained weights [w₀, w₁, w₂, ..., wₙ]
    """
    n_students = len(features)
    n_features = len(features[0])

    # Create binary labels
    binary_labels = [1 if label == target_house else 0 for label in labels]

    # Initialize weights
    weights = [0.0] * (n_features + 1)

    print(f"Training {target_house} vs Others (Mini-batch GD, batch_size={batch_size})...")

    # Mini-batch Gradient Descent
    for epoch in range(epochs):
        # Process data in mini-batches
        for batch_start in range(0, n_students, batch_size):
            batch_end = min(batch_start + batch_size, n_students)

            # Accumulate gradients for this mini-batch
            gradient_bias = 0.0
            gradient_features = [0.0] * n_features

            # Process mini-batch
            for student_idx in range(batch_start, batch_end):
                student_features = features[student_idx]
                actual_label = binary_labels[student_idx]

                # 1. Predict probability
                predicted_prob = predict_probability(student_features, weights)

                # 2. Calculate error
                error = predicted_prob - actual_label

                # 3. Accumulate gradients
                gradient_bias += error
                for feat_idx, feature_value in enumerate(student_features):
                    gradient_features[feat_idx] += error * feature_value

            # 4. Update weights using mini-batch gradients
            batch_actual_size = batch_end - batch_start
            weights[0] -= learning_rate * (gradient_bias / batch_actual_size)
            for feat_idx in range(n_features):
                weights[feat_idx + 1] -= learning_rate * (gradient_features[feat_idx] / batch_actual_size)

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}")

    print(f"✓ Training complete for {target_house}")

    return weights


def train_one_vs_all(features, labels, target_house, learning_rate=0.01, epochs=1000, method='sgd', batch_size=32):
    """
    Train binary logistic regression classifier for one house vs all others

    Supports 3 optimization methods:
    - SGD: Stochastic Gradient Descent (mandatory, default)
    - Batch: Batch Gradient Descent (BONUS 3)
    - Mini-batch: Mini-batch Gradient Descent (BONUS 3)

    Args:
        features: List of [x₁, x₂, ...] for each student (normalized)
        labels: List of house names for each student
        target_house: House name to train for (e.g., 'Gryffindor')
        learning_rate: Step size for gradient descent (default: 0.01)
        epochs: Number of training iterations (default: 1000)
        method: 'sgd', 'batch', or 'minibatch' (default: 'sgd')
        batch_size: For mini-batch only (default: 32)

    Returns:
        list: Trained weights [w₀, w₁, w₂, ..., wₙ]
    """
    if method == 'batch':
        return train_one_vs_all_batch(features, labels, target_house, learning_rate, epochs)
    elif method == 'minibatch':
        return train_one_vs_all_minibatch(features, labels, target_house, learning_rate, epochs, batch_size)
    else:  # default: sgd
        return train_one_vs_all_sgd(features, labels, target_house, learning_rate, epochs)


def calculate_accuracy(features, labels, all_weights, houses):
    """
    Calculate training accuracy

    For each student:
    1. Calculate probability for each house
    2. Predict house with highest probability
    3. Compare with actual label

    Args:
        features: List of normalized features
        labels: List of actual house labels
        all_weights: Dictionary {house: weights}
        houses: List of house names

    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    correct = 0
    total = len(labels)

    for student_idx in range(total):
        student_features = features[student_idx]
        actual_house = labels[student_idx]

        # Predict probabilities for all houses
        probabilities = {}
        for house in houses:
            weights = all_weights[house]
            prob = predict_probability(student_features, weights)
            probabilities[house] = prob

        # Choose house with highest probability
        predicted_house = max(probabilities, key=probabilities.get)

        if predicted_house == actual_house:
            correct += 1

    accuracy = correct / total
    return accuracy


def save_weights(all_weights, means, stds, feature_names, houses, filename='weights.csv'):
    """
    Save trained weights to CSV file

    Format:
    house,bias,feature1_weight,feature2_weight,...
    Gryffindor,0.5,0.3,-0.2,...
    Hufflepuff,-0.3,0.1,0.4,...
    ...

    Also save normalization parameters (means, stds)

    Args:
        all_weights: Dictionary {house: weights}
        means: List of feature means
        stds: List of feature stds
        feature_names: List of feature names
        houses: List of house names
        filename: Output file path
    """
    print(f"\nSaving weights to {filename}...")

    with open(filename, 'w') as f:
        # Header
        header = 'house,bias,' + ','.join(feature_names)
        f.write(header + '\n')

        # Weights for each house
        for house in houses:
            weights = all_weights[house]
            line = house + ',' + ','.join(str(w) for w in weights)
            f.write(line + '\n')

        # Save normalization parameters
        f.write('\n# Normalization parameters\n')
        f.write('means,' + ','.join(str(m) for m in means) + '\n')
        f.write('stds,' + ','.join(str(s) for s in stds) + '\n')

    print(f"✓ Weights saved to {filename}")


def main():
    # Check usage and parse arguments
    method = 'sgd'  # default
    batch_size = 32  # default

    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <dataset.csv> [--method sgd|batch|minibatch] [--batch-size N]")
        print("\nOptions:")
        print("  --method sgd         Stochastic Gradient Descent (default, fastest)")
        print("  --method batch       Batch Gradient Descent (most stable)")
        print("  --method minibatch   Mini-batch Gradient Descent (balanced)")
        print("  --batch-size N       Batch size for mini-batch (default: 32)")
        sys.exit(1)

    filename = sys.argv[1]

    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--method' and i + 1 < len(sys.argv):
            method = sys.argv[i + 1].lower()
            if method not in ['sgd', 'batch', 'minibatch']:
                print(f"Error: Invalid method '{method}'. Use 'sgd', 'batch', or 'minibatch'")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--batch-size' and i + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[i + 1])
                if batch_size <= 0:
                    print("Error: Batch size must be positive")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid batch size '{sys.argv[i + 1]}'")
                sys.exit(1)
            i += 2
        else:
            print(f"Error: Unknown argument '{sys.argv[i]}'")
            sys.exit(1)

    print("=" * 80)
    print("LOGISTIC REGRESSION TRAINING")
    print(f"Optimization Method: {method.upper()}")
    if method == 'minibatch':
        print(f"Batch Size: {batch_size}")
    print("=" * 80)

    # Load data
    features, labels, feature_names, houses = load_training_data(filename)

    # Normalize features
    normalized_features, means, stds = normalize_features(features)

    print(f"\n{'=' * 80}")
    print("Training One-vs-All Classifiers")
    print(f"{'=' * 80}\n")

    # Train one classifier per house
    all_weights = {}

    for house in houses:
        weights = train_one_vs_all(
            normalized_features,
            labels,
            house,
            learning_rate=0.1,  # Adjust if needed
            epochs=500,         # Adjust if needed
            method=method,
            batch_size=batch_size
        )
        all_weights[house] = weights
        print()

    # Calculate training accuracy
    print(f"{'=' * 80}")
    print("Evaluating Model")
    print(f"{'=' * 80}\n")

    accuracy = calculate_accuracy(normalized_features, labels, all_weights, houses)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")

    # Save weights
    save_weights(all_weights, means, stds, feature_names, houses)

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
