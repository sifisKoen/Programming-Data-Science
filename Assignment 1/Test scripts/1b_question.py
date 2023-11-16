import numpy as np
import pandas as pd


# Insert the functions create_normalization and apply_normalization below (after the comments)
#
# Input to create_normalization:
# df: a dataframe (where the column names "CLASS" and "ID" have special meaning)
# normalizationtype: "minmax" (default) or "zscore"
#
# Output from create_normalization:
# df            - a new dataframe, where each numeric value in a column has been replaced by a normalized value
# normalization - a mapping (dictionary) from each column name to a triple, consisting of
#                ("minmax",min_value,max_value) or ("zscore",mean,std)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID"),
#         the other columns should remain unchanged
#
# Hint 3: Take a close look at the lecture slides on data preparation
#
def create_normalization(df, normalizationtype="minmax"):
    # Step 1. Copy the input dataframe.
    normalized_df = df.copy()

    # Step 2. Initialize a dictionary so to store the normalization parameters.
    normalization_map = {}

    # Step 3. Iterate through all the columns.
    for column in normalized_df.columns:
        # Step 4. Check if the type of the column is numeric (float64 or int64) and is not labeled as CLASS or ID.
        if normalized_df[column].dtype in ["float64", "int64"] and column not in ["CLASS", "ID"]:
            # Step 5. Check if the normalization type is min-max.
            if normalizationtype == "minmax":
                # Step 6. Apply min-max normalization.
                minimum_value = normalized_df[column].min()
                maximum_value = normalized_df[column].max()
                normalized_df[column] = (normalized_df[column] - minimum_value) / (maximum_value - minimum_value)
                # Step 7. Store the normalization parameters.
                normalization_map[column] = (normalizationtype, minimum_value, maximum_value)
            # Step 8. Check if the normalization type is z-normalization / zscore.
            elif normalizationtype == "zscore":
                # Step 9. Apply the z-normalization.
                mean_value = normalized_df[column].mean()
                std_value = normalized_df[column].std()
                normalized_df[column] = (normalized_df[column] - mean_value) / std_value
                # Step 10. Store the normalization parameters.
                normalization_map[column] = (normalizationtype, mean_value, std_value)
            # Step 11. Handle unexpected normalization type
            else:
                raise TypeError("Unsupported normalization type!")
            
    return normalized_df, normalization_map



glass_train_df = pd.read_csv("../glass_train.csv")

glass_test_df = pd.read_csv("../glass_test.csv")

glass_train_norm, normalization = create_normalization(glass_train_df,normalizationtype="minmax")
print("normalization:\n")
for f in normalization:
    print("{}:{}".format(f,normalization[f]))

print()