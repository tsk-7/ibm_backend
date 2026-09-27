# Normalization

Normalization rescales numeric features to a bounded range, commonly 0 to 1. Min-max scaling uses (x - minimum) / (maximum - minimum). It can help distance-based methods when features have different units, but an extreme value can compress most observations into a narrow range.

Fit scaling parameters on training data only when building predictive models, then reuse them for later data. Normalization changes units and is not generally required for tree-based methods. Inspect the current distribution and intended analysis before applying it.