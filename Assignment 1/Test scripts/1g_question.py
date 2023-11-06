import pandas as pd
import numpy as np


# Insert the function accuracy below
#
# Input to accuracy:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from accuracy:
# accuracy - the fraction of cases for which the predicted class label coincides with the correct label
#
# Hint: In case the label receiving the highest probability is not unique, you may
#       resolve that by picking the first (as ordered by the column names) or 
#       by randomly selecting one of the labels with highest probaility.
def accuracy(df: pd.DataFrame, correctlabels: list):

    if len(df) != len(correctlabels):
        raise ValueError("The number of rows in the DataFrame must equal the number of correct labels")
    
    # Step 2: Find the label with the highest probability in each row
    predict_labels = df.idxmax(axis=1)

    # Step 3: Compare with the correct labels
    matches = predict_labels == correctlabels

    # Step 4: Calculate accuracy 
    accuracy = matches.sum() / len(correctlabels)

    return accuracy


predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})
print("predictions",predictions)

correctlabels = ["B","A","B","B","C"]

print("Accuracy: {}".format(accuracy(predictions,correctlabels))) # Note that depending on how ties are resolved the accuracy may be 0.6 or 0.8