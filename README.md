# DSLR - Data Science Logistic Regression

Hogwarts house classification using logistic regression from scratch (no ML libraries).

## 📂 Project Structure

```
DSLR/
├── README.md                     # Project documentation
├── DSLR.pdf                      # Project specification
│
├── datasets/                     # Training and test data
│   ├── dataset_train.csv        # 1600 students with labels
│   └── dataset_test.csv         # 400 students without labels
│
├── srcs/                    # Required implementations (includes BONUS features)
│   ├── describe.py              # Statistical analysis (8 basic + 7 BONUS metrics)
│   ├── histogram.py             # Score distribution visualization
│   ├── scatter_plot.py          # Correlation analysis
│   ├── pair_plot.py             # Feature relationship matrix
│   ├── logreg_train.py          # Model training (SGD + BONUS: Batch/Mini-batch GD)
│   └── logreg_predict.py        # House prediction
│
└── outputs/                      # Generated results
    ├── visualizations/          # PNG images
    └── models/                  # Trained weights
```

## 🚀 Usage

### Basic Usage (srcs)

#### 1. Data Analysis
```bash
cd srcs/

# Statistical summary (shows all 15 metrics including BONUS)
python describe.py ../datasets/dataset_train.csv

# Visualizations
python histogram.py ../datasets/dataset_train.csv
python scatter_plot.py ../datasets/dataset_train.csv
python pair_plot.py ../datasets/dataset_train.csv
```

#### 2. Model Training & Prediction
```bash
# Train model (default: SGD method)
python logreg_train.py ../datasets/dataset_train.csv

# Make predictions
python logreg_predict.py ../datasets/dataset_test.csv ../outputs/models/weights.csv
```

### BONUS Features

#### BONUS 1: Extended Statistics (in describe.py)
The `describe.py` file includes **7 additional statistical metrics**:
- **Median**: 50th percentile (middle value)
- **Mode**: Most frequent value
- **Range**: Max - Min
- **IQR**: Interquartile Range (Q3 - Q1)
- **Variance**: Standard deviation squared
- **Skewness**: Distribution asymmetry measure
- **Kurtosis**: Distribution tail heaviness

All automatically calculated when running:
```bash
python srcs/describe.py datasets/dataset_train.csv
```

#### BONUS 2 & 3: Multiple Gradient Descent Methods (in logreg_train.py)
The `logreg_train.py` file supports **3 optimization methods**:

```bash
# Default: Stochastic Gradient Descent (srcs)
python srcs/logreg_train.py datasets/dataset_train.csv

# BONUS: Batch Gradient Descent
python srcs/logreg_train.py datasets/dataset_train.csv --method batch

# BONUS: Mini-batch Gradient Descent
python srcs/logreg_train.py datasets/dataset_train.csv --method minibatch --batch-size 32
```

**Performance Comparison:**
| Method | Description | Training Time | Accuracy |
|--------|-------------|---------------|----------|
| **SGD** | Updates per sample (srcs) | 3.52s | 98.32% |
| **Batch GD** | Updates per epoch (BONUS) | 3.10s ⚡ | 98.32% |
| **Mini-batch GD** | Updates per batch (BONUS) | 3.18s | 98.32% |

## 📊 Implementation Details

### Phase 1-4: Data Analysis & Visualization

**describe.py**: Statistical analysis from scratch
- **Basic (srcs)**: Count, Mean, Std, Min, 25%, 50%, 75%, Max
- **BONUS**: Median, Mode, Range, IQR, Variance, Skewness, Kurtosis

**histogram.py**: Distribution analysis
- Overlaid histograms for 4 houses
- Identifies most homogeneous subject

**scatter_plot.py**: Correlation analysis
- Analyzes all 78 subject pairs
- Pearson correlation coefficient
- Identifies most similar features

**pair_plot.py**: Multi-feature relationships
- Diagonal: Distribution histograms
- Off-diagonal: Scatter plots
- Automatic selection of 6 most significant features

### Phase 5-6: Logistic Regression

**logreg_train.py**: Model training
- One-vs-All multi-class strategy
- **srcs**: SGD (Stochastic Gradient Descent)
- **BONUS**: Batch GD and Mini-batch GD
- Normalization and weight export

**logreg_predict.py**: Prediction
- Load trained weights
- Apply same normalization
- Predict houses for test data

## 🎯 Key Concepts

### Statistics (BONUS 1)
- **Mean/Std/Min/Max**: Basic statistics (srcs)
- **Percentiles**: 25%, 50%, 75% (srcs)
- **Median**: Middle value = 50th percentile (BONUS)
- **Mode**: Most frequent value (BONUS)
- **Range**: Spread of data (BONUS)
- **IQR**: Interquartile range for outlier detection (BONUS)
- **Variance**: Squared standard deviation (BONUS)
- **Skewness**: Distribution asymmetry (-: left, +: right) (BONUS)
- **Kurtosis**: Tail heaviness (-: light, +: heavy) (BONUS)

### Gradient Descent Variants (BONUS 2 & 3)

**Stochastic Gradient Descent (SGD)** - srcs
- Updates weights after **each sample**
- Fast, escapes local minima
- High variance, noisy convergence

**Batch Gradient Descent** - BONUS 3
- Updates weights after **all samples**
- Stable, smooth convergence
- Slower for large datasets

**Mini-batch Gradient Descent** - BONUS 3
- Updates weights after **small batches**
- Balance between speed and stability
- Requires batch_size tuning

### Machine Learning
- **Logistic Regression**: Binary classification with sigmoid
- **One-vs-All**: Multi-class classification strategy
- **Sigmoid Function**: Maps values to (0, 1)
- **Normalization**: Feature scaling (mean=0, std=1)

## 📝 Bonus Implementation Notes

### How BONUS Features are Organized

All BONUS features are integrated into the srcs files with clear markers:

**In describe.py:**
```python
# ==================== BONUS: Additional Statistical Functions ====================
def calculate_median(column):
    """Calculate median (50th percentile) - Middle value when data is sorted"""
    ...
```

**In logreg_train.py:**
```python
def train_one_vs_all_batch(features, labels, target_house, ...):
    """
    BONUS 3: Train using Batch Gradient Descent
    Updates weights after processing ALL samples
    ...
    """
```

This approach:
- ✅ Keeps codebase clean and unified
- ✅ Clearly marks BONUS features with comments
- ✅ Makes it easy for evaluators to identify optional features
- ✅ Allows running basic version (no flags) or BONUS version (with flags)

## 🎓 Learning Outcomes

1. ✅ Understanding of basic and advanced statistics
2. ✅ Data visualization techniques
3. ✅ Logistic regression mathematics
4. ✅ Multiple gradient descent optimization strategies
5. ✅ Multi-class classification with One-vs-All
6. ✅ Model evaluation and validation

## 📈 Final Results

- **Training Accuracy**: 98.32% (all methods)
- **Features Used**: 13 subjects
- **Classes**: 4 Hogwarts houses
- **Training Samples**: ~1600 students
- **Test Samples**: 400 students

## 📚 Summary

**srcs Features (All Implemented):**
- ✅ Phase 1-4: Data analysis and visualization
- ✅ Phase 5-6: Logistic regression with SGD

**BONUS Features (All Implemented):**
- 🌟 **BONUS 1**: 7 additional statistics in describe.py
- 🌟 **BONUS 2**: Stochastic GD (already implemented as srcs)
- 🌟 **BONUS 3**: Batch GD + Mini-batch GD in logreg_train.py

**Total Statistics**: 15 metrics (8 basic + 7 BONUS)
**Total GD Methods**: 3 variants (SGD srcs + 2 BONUS)

---

**Author**: 42 School DSLR Project
**Date**: November 2024
**Status**: All srcs + all bonus features implemented ✅
**Implementation**: Pure Python (no numpy, pandas, scikit-learn)
