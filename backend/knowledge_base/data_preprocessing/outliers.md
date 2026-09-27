# Outliers

An outlier is an observation unusually distant from the rest under a chosen rule; it may be an error, a rare valid event, or a meaningful signal. The IQR rule flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where IQR = Q3 - Q1.

Investigate source records and domain limits before removing or capping observations. The IQR flag is a screening rule, not proof of invalidity. DataInsight reports numerical outliers with the IQR rule and leaves decisions to the user.