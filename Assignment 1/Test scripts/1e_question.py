import pandas as pd
import numpy as np

# Insert the functions create_one_hot and apply_one_hot below
#
# Input to create_one_hot:
# df: a dataframe
#
# Output from create_one_hot:
# df      - a new dataframe, where each categoric feature has been replaced by a set of binary features 
#           (as many new features as there are possible values)
# one_hot - a mapping (dictionary) from column name to a set of categories (possible values for the feature)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider columns of type "object" or "category" only (and which are not labeled "CLASS" or "ID")
#
# Hint 3: Consider creating new column names by merging the original column name and the categorical value
#
# Hint 4: Set all new columns to be of type "float"
#
# Hint 5: Do not forget to remove the original categoric feature
#
def create_one_hot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    # Step 1: Copy the input dataframe to avoid changes to the initial dataframe
    copy_df = df.copy()
    # Step 2: Initialize a dictionary to stor one-hot information
    one_hot = {}

    # Step 3: Iterate through the columns of the dataframe and apply the one-hot encoding to categorical columns
    for column in copy_df.columns:
        if column not in ["CLASS", "ID"] and copy_df[column].dtype in ["category", "object"]:

            # Get unique categories for the columns
            unique_column_category = df[column].unique()
            # Store the unique categories into dictionary
            one_hot[column] = unique_column_category
            # Generate one-hot encoding
            for category in sorted(unique_column_category):
                # Create a new column for each category
                new_column_category_name = f"{column}_{category}"
                copy_df[new_column_category_name] = (df[column] == category).astype(float)

            # Remove the original column
            copy_df.drop(column, axis=1, inplace=True)

    return copy_df, one_hot
# Input to apply_one_hot:
# df      - a dataframe
# one_hot - a mapping (dictionary) from column name to categories
#
# Output from apply_one_hot:
# df - a new dataframe, where each categoric feature has been replaced by a set of binary features
#
# Hint: See the above Hints
def apply_one_hot(df: pd.DataFrame, one_hot: dict) -> pd.DataFrame:

    # Step 1: Copy the input dataframe to avoid changes to the initial dataframe
    copy_df = df.copy()

    # Step 2: Iterate through the one-hot dictionary so to apply the one-hot encoding
    for column, categories in one_hot.items():
        if column in df.columns:
            for category in categories:
                new_column_category_name = f"{column}_{category}"
                # Create a new column an we add 0 as the default value
                copy_df[new_column_category_name] = 0
                # We add 1 if the category matches 
                copy_df.loc[df[column] == category, new_column_category_name] = 1
            # Remove the original value
            copy_df.drop(column, axis=1, inplace=True)

    return copy_df

train_df = pd.read_csv("../tic-tac-toe_train.csv")

new_train, one_hot = create_one_hot(train_df)
test_df = pd.read_csv("../tic-tac-toe_test.csv")
new_test_df = apply_one_hot(test_df,one_hot)
print("new_test_df",new_test_df)