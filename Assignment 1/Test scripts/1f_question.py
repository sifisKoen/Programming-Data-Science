import pandas as pd
import numpy as np

# Insert the function split below
#
# Input to split:
# df           - a dataframe
# testfraction - a float in the range (0,1) (default = 0.5)
#
# Output from split:
# trainingdf - a dataframe consisting of a random sample of (1-testfraction) of the rows in df
# testdf     - a dataframe consisting of the rows in df that are not included in trainingdf
#
# Hint: You may use np.random.permutation(df.index) to get a permuted list of indexes where a 
#       prefix corresponds to the test instances, and the suffix to the training instances 
def split(df: pd.DataFrame, testfraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # Step 1: Check if the testfraction is between 0 and 1
    if not (0 < testfraction < 1):
        raise ValueError("testfraction must be between 0 and 1")
    
    # Step 2: Shuffle the indices
    shuffle_indices = np.random.permutation(df.index)

    # Step 3; Calculate the size of the test instances
    test_set_size = int(len(df) * testfraction)

    # Step 4: Split the indicates into test and train set
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    
    # Step 4: Create the test and the train dataframes
    training_df = df.loc[train_indices]
    test_df  = df.loc[test_indices]

    return training_df, test_df
# Test your code  (leave this part unchanged)

glass_df = pd.read_csv("../glass.csv")

glass_train, glass_test = split(glass_df,testfraction=0.25)

print("Training IDs:\n{}".format(glass_train["ID"].values))

print("\nTest IDs:\n{}".format(glass_test["ID"].values))

print("\nOverlap: {}".format(set(glass_train["ID"]).intersection(set(glass_test["ID"]))))
