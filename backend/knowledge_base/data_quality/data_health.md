# Data Health

Data health combines measurable quality dimensions into a review signal; it does not certify that data is correct or fit for every purpose. DataInsight reports completeness from missing cells, consistency from duplicate rows, and validity from numerical IQR outlier cells, then averages those three component scores.

Read the returned formulas and component percentages together. The score is deterministic for the current dataset state; investigate the underlying missingness, duplicates, and flagged outliers before changing data.