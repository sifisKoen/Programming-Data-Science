import pandas as pd
import numpy as np


# Insert the functions create_imputation and apply_imputation below (after the comments)
#
# Input to create_imputation:
# df: a dataframe (where the column names "CLASS" and "ID" have special meaning)
#
# Output from create_imputation:
# df         - a new dataframe, where each missing numeric value in a column has been replaced by the mean of that column 
#              and each missing categoric value in a column has been replaced by the mode of that column
# imputation - a mapping (dictionary) from column name to value that has replaced missing values
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Handle columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID") in one way
#         and columns of type "object" and "category" in other ways
#
# Hint 3: Consider using the pandas functions mean and mode respectively, as well as fillna
#
# Hint 4: In the rare case of all values in a column being missing*, replace numeric values with 0,
#         object values with "" and category values with the first category (cat.categories[0])  
#
#         *Note that this will not occur if the previous column filter function has been applied
#
def create_imputation(df):

    # Step 1: Copy the input dataframe
    copy_df = df.copy()

    # Strep 2: Initialize a dictionary to store imputation values
    imputation = {}

    # Step 3: Iterate trough all the columns in the dataframe
    for column in copy_df.columns:

        # Step 4: Skip the "CLASS" and "ID" columns
        if column not in ["CLASS", "ID"]:

            # Step 5: Handle numeric columns (float and int types)
            if np.issubdtype(copy_df[column].dtype, np.number): # = if copy_df[column].dtype in ["float64", "int64"]
                # Calculate the mean value of the column
                mean_value = copy_df[column].mean()
                # Check if all the values of the column are missing and replace mean with 0
                if pd.isnull(mean_value):
                    mean_value = 0
                # Replace the missing values with the mean value
                copy_df[column].fillna(mean_value, inplace=True)
                # Add the mean value to the imputation dictionary
                imputation[column] = mean_value

            # Step 6: Handle the object and the category columns
            elif copy_df[column].dtype in ["object", "category"]:
                # Calculate the mode of the column (Find the most frequent value appear in the column)
                mode_value = copy_df[column].mode().iloc[0] if not copy_df[column].mode().empty else ""
                # If all values are missing and column is category then replace the mode value with the first category.
                # If no categories it will raise an error.
                if copy_df[column].dtype.name == "category" and mode_value == "":               
                    mode_value = copy_df[column].cat.categories[0]
                # Replace missing values with the mode value
                copy_df[column].fillna(mode_value, inplace=True)
                # Add the mode value to the imputation dictionary
                imputation[column] = mode_value

    return copy_df, imputation

# Input to apply_imputation:
# df         - a dataframe
# imputation - a mapping (dictionary) from column name to value that should replace missing values
#
# Output from apply_imputation:
# df - a new dataframe, where each missing value has been replaced according to the mapping
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider using fillna
def apply_imputation(df, imputation):
    # Step 1: Copy the input dataframe
    copy_df = df.copy()
    # Step 2: Iterate through the imputation dictionary and take the items of it
    for column_name, value in imputation.items():
        # Fill the empty values of the copied dataframe according the dictionary map.
        copy_df[column_name].fillna(value, inplace=True)

    return copy_df
        


anneal_train_df = pd.read_csv("../anneal_train.csv")
anneal_test_df = pd.read_csv("../anneal_test.csv")

anneal_train_imp, imputation = create_imputation(anneal_train_df)
anneal_test_imp = apply_imputation(anneal_test_df,imputation)

print("Imputation:\n")
for f in imputation:
    print("{}:{}".format(f,imputation[f]))

print("\nNo. of replaced missing values in training data:\n{}".format(anneal_train_imp.count()-anneal_train_df.count()))
print("\nNo. of replaced missing values in test data:\n{}".format(anneal_test_imp.count()-anneal_test_df.count()))