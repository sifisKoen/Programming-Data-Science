import pandas as pd
import numpy as np

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
def create_normalization(df, normalizationtype = "minmax"):


    normalized_df = glass_train_df.copy()

    normalization_map = {}

    for column in normalized_df.columns:
        if normalized_df[column].dtype in ["float64", "int64"] and column not in ["CLASS", "ID"]:
            if normalizationtype == "minmax":
                minimum_value = normalized_df[column].min()
                maximum_value = normalized_df[column].max()
                normalized_df[column] = (normalized_df[column] - minimum_value) / maximum_value - minimum_value

                normalization_map[column] = (normalizationtype, minimum_value, maximum_value)
            elif normalizationtype == "zscore":
                mean_value = normalized_df[column].mean()
                std_value = normalized_df[column].std()

                normalized_df[column] = (normalized_df[column] - mean_value) / std_value

                normalization_map[column] = (normalizationtype, mean_value, std_value)

    return normalized_df, normalization_map

#
# Input to apply_normalization:
# df            - a dataframe
# normalization - a mapping (dictionary) from column names to triples (see above)
#
# Output from apply_normalization:
# df - a new dataframe, where each numerical value has been normalized according to the mapping
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: For minmax-normalization, you may consider to limit the output range to [0,1]
def apply_normalization(df, normalization):
    
    copy_df = df.copy()

    for column, values in normalization.items():

        if column in copy_df.columns:

            normalization_type, first_variable, second_variable = values

        if normalization_type == "minmax":

            copy_df[column] = (copy_df[column] - first_variable) / (second_variable - first_variable)

            # Find documentation here: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
            copy_df[column] = np.clip(copy_df[column], 0, 1)
        elif normalization_type == "zscore":

            copy_df[column] = (copy_df[column] - first_variable) / second_variable
        else:
            raise TypeError("Unsupported normalization type!")



    return copy_df



glass_train_df = pd.read_csv("glass_train.csv")

glass_test_df = pd.read_csv("glass_test.csv")

glass_train_norm, normalization = create_normalization(glass_train_df,normalizationtype="minmax")
print("normalization:\n")
for f in normalization:
    print("{}:{}".format(f,normalization[f]))

print()
    
glass_test_norm = apply_normalization(glass_test_df,normalization)
print("glass_test_norm",glass_test_norm)

