# Duplicate Records and Quality

Duplicates can distort record counts and downstream summaries when each row is expected to represent one unique observation. Exact row equality is only one definition: entity-level duplicates may require a key and a rule for resolving conflicting records.

Measure before removal, confirm the intended uniqueness rule, preserve recoverability, and verify after the operation. Repeated events can be valid and should not be discarded merely because values repeat.