import pandas as pd
import numpy as np

# Insert the function auc below
#
# Input to auc:
# df            - a dataframe with class labels as column names and each row corresponding to
#                 a prediction with estimated probabilities for each class
# correctlabels - an array (or list) of the correct class label for each prediction
#                 (the number of correct labels must equal the number of rows in df)
#
# Output from auc:
# auc - the weighted area under ROC curve
#
# Hint 1: Calculate the binary AUC first for each class label c, i.e., treating the
#         predicted probability of this class for each instance as a score; the true positives
#         are the ones belonging to class c and the false positives the rest
#
# Hint 2: When calculating the binary AUC, first find the scores of the true positives and then
#         the scores of the true negatives
#
# Hint 3: You may use a dictionary with a mapping from each score to an array of two numbers; 
#         the number of true positives with this score and the number of true negatives with this score
#
# Hint 4: Created a (reversely) sorted (on the scores) list of pairs from the dictionary and
#         iterate over this to additively calculate the AUC
#
# Hint 5: For each pair in the above list, there are three cases to consider; the no. of false positives
#         is zero, the no. of true positives is zero, and both are non-zero
#
# Hint 6: Calculate the weighted AUC by summing the individual AUCs weighted by the relative
#         frequency of each class (as estimated from the correct labels)
import pandas as pd
import numpy as np

def auc(df, correctlabels):
    # Helper function to calculate binary AUC for a single class
    def binary_auc(class_name):
        scores = df[class_name].values
        true_positives = [int(label == class_name) for label in correctlabels]
        false_positives = [int(label != class_name) for label in correctlabels]

        # Create triples of (score, tp, fp)
        triples = [(score, tp, fp) for score, tp, fp in zip(scores, true_positives, false_positives)]

        # Sort the triples based on scores in reverse order
        triples.sort(key=lambda x: x[0], reverse=True)

        # Calculate AUC using the lecture's algorithm
        auc = 0
        cov_tp = 0
        tot_tp = sum(true_positives)
        tot_fp = sum(false_positives)

        for _, tp, fp in triples:
            if fp == 0:
                cov_tp += tp
            elif tp == 0:
                auc += (cov_tp / tot_tp) * (fp / tot_fp)
            else:
                auc += (cov_tp / tot_tp) * (fp / tot_fp) + (tp / tot_tp) * (fp / tot_fp) / 2
                cov_tp += tp

        return auc

    # Calculate binary AUC for each class and store the results
    auc_results = {class_name: binary_auc(class_name) for class_name in df.columns}

    # Calculate weighted AUC
    class_counts = pd.Series(correctlabels).value_counts(normalize=True)
    
    weighted_auc = sum(auc_results[class_name] * class_counts.get(class_name, 0) for class_name in df.columns)

    return weighted_auc

predictions = pd.DataFrame({"A":[0.9,0.9,0.6,0.55],"B":[0.1,0.1,0.4,0.45]})

correctlabels = ["A","B","B","A"]

print("AUC: {}".format(auc(predictions,correctlabels)))

predictions = pd.DataFrame({"A":[0.5,0.5,0.5,0.25,0.25],"B":[0.5,0.25,0.25,0.5,0.25],"C":[0.0,0.25,0.25,0.25,0.5]})

correctlabels = ["B","A","B","B","C"]

print("AUC: {}".format(auc(predictions,correctlabels)))