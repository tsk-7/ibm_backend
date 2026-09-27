# Missing Values

Missing values represent unknown, unavailable, or inapplicable observations. First measure their count and percentage by column, then determine whether missingness is random, related to other fields, or structurally expected. Do not treat a blank as zero unless the domain definition explicitly says zero is correct.

Common choices are removing rows when the loss is small and unbiased, using mean or median for suitable numeric columns, using mode for categorical fields, or carrying values forward/backward only for ordered observations. Median is less sensitive to extreme values than mean. Record the method because imputation changes the data.