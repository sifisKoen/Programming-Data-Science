import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.tree import DecisionTreeClassifier

# Copy and paste functions from Assignment 1 here that you need for this assignment

# COLUMN FILTER

def create_column_filter(df):

    # Step 1. Copy the input dataframe.
    copy_df = df.copy()

    # Step 2. Initialise an empty list so to store the names of the columns witch will be kept.
    column_filter = []

    # Step 3. Iterate through all the columns in the dataframe.
    for column in copy_df.columns:

        # Step 4. Check if the column name is CLASS or ID so do not drop them.
        if column == "CLASS" or column == "ID":
            column_filter.append(column)
        else:

            # Step 5. Drop missing values and get unique values.
            unique_values = copy_df[column].dropna().unique()
            
            # Step 6. Check if the number of unique (non-missing) values is more than one, keep the column
            if (len(unique_values) > 1):
                column_filter.append(column)
            else:
                # Step 7. Drop the column from the dataframe copy.
                copy_df.drop(columns=[column], inplace=True)
        
        
    # Return the modified dataframe and the list of remaining columns
    return copy_df, column_filter

def apply_column_filter(df, column_filter):

    # Step 1. Copy the initial dataframe.
    copy_df = df.copy()

    # Step 2. Iterate through all the columns of the copied dataframe.
    for column in copy_df.columns:

        # Step 3. Check the column name if it is in the list of the filtered columns.  
        if column not in column_filter:
            # Step 4. If the column name is not in the filtered columns then it will be dropped.
            copy_df.drop(columns=[column], inplace=True)

    # Step 5. Return the modified dataframe with only the columns witch are in the filtered columns list
    return copy_df

# IMPUTATION

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

def apply_imputation(df, imputation):
    # Step 1: Copy the input dataframe
    copy_df = df.copy()
    # Step 2: Iterate through the imputation dictionary and take the items of it
    for column_name, value in imputation.items():
        # Fill the empty values of the copied dataframe according the dictionary map.
        copy_df[column_name].fillna(value, inplace=True)

    return copy_df

# ONE-HOT ENCODING

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

# ACCURACY

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

# AUC

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

# BRIER_SCORE

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


class RandomForest:
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None


    def fit(self, df, no_trees=100):
        copy_df = df.copy()

        # Create all necessary mappings and store them in the self object
        copy_df, self.column_filter = create_column_filter(copy_df)
        copy_df, self.imputation = create_imputation(copy_df)
        copy_df, self.one_hot = create_one_hot(copy_df)

        # Get the unique labels from "CLASS" column and sort them
        self.labels = sorted(copy_df["CLASS"].astype("category").cat.categories)


        # Drop CLASS column and turn into a numpy array
        features = copy_df.drop(columns=["CLASS"]).to_numpy()

        random_forest = []
        # Create a random forest with size of no_trees
        for i in range(no_trees):
            # Create a bootstrap sample using np.random.choice to randomly replace some values with row numbers of the ndarray
            bootstrap_sample = np.random.choice(len(features), len(features), replace=True)

            # Generate a decision tree using the DecisionTreeClassifier and the number of evaluated features is log2|F|
            decision_tree = DecisionTreeClassifier(max_features="log2")
            
            # Fit the decision tree using the bootstrap sample and the labels
            decision_tree.fit(features[bootstrap_sample], copy_df['CLASS'].to_numpy()[bootstrap_sample])

            # Add each decision tree to the random forest
            random_forest.append(decision_tree)

        # Add the decision tree to the model which will end up as a RandomForest
        self.model = random_forest

    def predict(self, df):
            copy_df = df.copy()

            # Drop column "CLASS" and "ID" from the df
            copy_df.drop(columns=["CLASS"], inplace=True)

            # Apply all necessary mapping that were created in the fit method to the df
            copy_df = apply_column_filter(copy_df, self.column_filter)
            copy_df = apply_imputation(copy_df, self.imputation)
            copy_df = apply_one_hot(copy_df, self.one_hot)

            # Create a matrix filled with zeros containing one row for each test instance and one column for each class label
            prediction_matrix = np.zeros((len(copy_df), len(self.labels)))

            # Create a mapping from class labels to indices
            label_to_index = {label: index for index, label in enumerate(self.labels)}

            for tree in self.model:
                # Get the prediction of the values for each tree in the RandomForest
                prediction = tree.predict_proba(copy_df.values)

                # For each column output by predict_proba
                for i, label in enumerate(tree.classes_):
                    # Find the index of this label in the overall class labels
                    index = label_to_index[label]
                    # Add the prediction to the appropriate column in the prediction matrix
                    prediction_matrix[:, index] += prediction[:, i]

            # Get the average prediction of all trees
            prediction_matrix = prediction_matrix / len(self.model)

            # Create a dataframe with the average predictions of all trees from the prediction matrix
            predictions = pd.DataFrame(prediction_matrix, columns=self.labels)

            return predictions
    
# Test your code (leave this part unchanged, except for if auc is undefined)

train_df = pd.read_csv("../anneal_train.csv")

test_df = pd.read_csv("../anneal_test.csv")

rf = RandomForest()

t0 = time.perf_counter()
rf.fit(train_df)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

test_labels = test_df["CLASS"]

t0 = time.perf_counter()
predictions = rf.predict(test_df)
print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))

print("Accuracy: {:.4f}".format(accuracy(predictions,test_labels)))
print("AUC: {:.4f}".format(auc(predictions,test_labels))) # Comment this out if not implemented in assignment 1
print("Brier score: {:.4f}".format(brier_score(predictions,test_labels))) # Comment this out if not implemented in assignment 1