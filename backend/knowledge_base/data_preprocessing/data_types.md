# Data Types

Data types describe how values should be interpreted and processed. Numeric-looking identifiers may be categorical strings; dates should be parsed as dates; booleans should have a controlled mapping; and numeric measures should be numeric for summaries and charts.

Validate conversion failures instead of silently coercing unexpected values to missing. Check representative values, null behavior, and downstream calculations after conversion. DataInsight supports integer, float, string, boolean, and datetime conversions through its preprocessing service.