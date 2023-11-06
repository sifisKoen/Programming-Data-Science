import pandas as pd
import numpy as np

# Insert the functions create_bins and apply_bins below
#
# Input to create_bins:
# df      - a dataframe
# nobins  - no. of bins (default = 10)
# bintype - either "equal-width" (default) or "equal-size" 
#
# Output from create_bins:
# df      - a new dataframe, where each numeric feature value has been replaced by a categoric (corresponding to some bin)
# binning - a mapping (dictionary) from column name to bins (threshold values for the bin)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Discretize columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID")
#
# Hint 3: Consider using pd.cut and pd.qcut respectively, with labels=False and retbins=True
#
# Hint 4: Set all columns in the new dataframe to be of type "category"
#
# Hint 5: Set the categories of the discretized features to be [0,...,nobins-1]
#
# Hint 6: Change the first and the last element of each binning to -np.inf and np.inf respectively 
#
def create_bins(df: pd.DataFrame, nobins: int, bintype: str) -> tuple[pd.DataFrame, dict]:

    # Step 1: Copy the input dataframe to avoid changes to the initial dataframe
    copy_df = df.copy()
    # Step 2: Initialize a dictionary to stor the binning information
    binning = {}

    # Step 3: Iterate the through all column of the dataframe
    for column in copy_df.columns:
        # Check if the the column name is "CLASS" or "ID" and if the type of the column is numeric ("float64", "int64"])
        if column not in ["CLASS", "ID"] and copy_df[column].dtype in ["float64", "int64"]:
            # Apply equal-width binning
            if bintype == "equal-width":
                copy_df[column], bins =  pd.cut(copy_df[column], nobins, labels=False, retbins=True)
            # Apply the equal-size binning
            elif bintype == "equal-size":
                copy_df[column], bins = pd.qcut(copy_df[column], nobins, retbins=True, labels=False, duplicates="drop")

            # Step 4: Adjust the firs and the last bin to the -infinity and +infinity respectively
            bins[0], bins[-1] = -np.inf, np.inf
            # Step 5: Stor the binning information to the dictionary
            binning[column] = bins
            # Step 6: Convert the columns into the categorical data type
            copy_df[column] = pd.Categorical(copy_df[column], categories=range(nobins))

    return copy_df, binning

# Input to apply_bins:
# df      - a dataframe
# binning - a mapping (dictionary) from column name to bins (threshold values for the bin)
#
# Output from apply_bins:
# df - a new dataframe, where each numeric feature value has been replaced by a categoric (corresponding to some bin)
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider using pd.cut 
#
# Hint 3: Set all columns in the new dataframe to be of type "category"
#
# Hint 4: Set the categories of the discretized features to be [0,...,nobins-1]
def apply_bins(df: pd.DataFrame, binning: dict) -> pd.DataFrame:

    # Step 1: Copy the input dataframe to avoid changes to the initial dataframe
    copy_df = df.copy()

    # Step 2: Iterate through binning dictionary
    for column, bins in binning.items():
        # Apply binning to the columns that a in the binning dictionary
        if column in copy_df.columns:
            # Discretize the columns based on the bins
            copy_df[column] = pd.cut(copy_df[column], bins, labels=False)
            # Convert the column into the categorical data type 
            copy_df[column] = pd.Categorical(copy_df[column], categories=range(len(bins) - 1))

    return copy_df



glass_train_df = pd.read_csv("../glass_train.csv")

glass_test_df = pd.read_csv("../glass_test.csv")

glass_train_disc, binning = create_bins(glass_train_df,nobins=10,bintype="equal-size")
print("binning:")
for f in binning:
    print("{}:{}".format(f,binning[f]))

print()    
glass_test_disc = apply_bins(glass_test_df,binning)
print("glass_test_disc",glass_test_disc)
