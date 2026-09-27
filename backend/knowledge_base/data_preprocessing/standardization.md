# Standardization

Standardization converts a numeric feature to a z-score: (value - mean) / standard deviation. The transformed feature has mean near zero and standard deviation near one when calculated on the same sample. Unlike min-max scaling, values are not restricted to a fixed interval.

Standardization is often useful for distance-based, gradient-based, and regularized methods that are sensitive to feature scale. It can be affected by outliers and is not usually needed for decision trees. Estimate mean and standard deviation from training data only in predictive workflows.