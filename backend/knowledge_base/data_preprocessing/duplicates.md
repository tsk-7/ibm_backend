# Duplicate Rows

Duplicate rows can inflate counts, averages, and model importance. Confirm whether rows are exact duplicates or repeated entities that should be unique only by selected key columns. Repeated transactions can be valid, so deduplication needs a domain rule.

Before removal, record the duplicate count and retain a recoverable original or backup. After removal, verify the remaining duplicate count and compare row totals. DataInsight detects exact duplicate rows and its remove action keeps the first occurrence.