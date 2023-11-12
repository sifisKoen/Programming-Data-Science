# Insert the function brier_score below
#
# Input to brier_score:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from brier_score:
# brier_score - the average square error of the predicted probabilties 
#
# Hint: Compare each predicted vector to a vector for each correct label, which is all zeros except 
#       for at the index of the correct class. The index can be found using np.where(df.columns==l)[0] 
#       where l is the correct label.

import pandas as pd
import numpy as np

def brier_score(df, correctlabels):
    total_brier_score = 0.0

    for i in range(len(df)):
        pred = df.iloc[i].values  
        corr_label = correctlabels[i] 
        
        # Find the index of the correct label in the columns of the dataframe
        corr_label_i = np.where(df.columns == corr_label)[0][0]
        corr_vector = np.zeros(len(pred))
        corr_vector[corr_label_i] = 1
        squared_error = np.sum((pred - corr_vector) ** 2)
        total_brier_score += squared_error
    
    avg_brier_score = total_brier_score / len(df)
    
    return avg_brier_score

# Test your code  (leave this part unchanged)

predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

correctlabels = ["B","A","B","B","C"]

print("Brier score: {}".format(brier_score(predictions,correctlabels)))