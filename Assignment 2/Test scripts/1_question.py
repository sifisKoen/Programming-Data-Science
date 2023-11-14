import numpy as np
import pandas as pd
import time
from scipy.spatial import distance


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

def create_normalization(df, normalizationtype = "minmax"):

    # Step 1. Copy the input dataframe.
    normalized_df = df.copy()

    # Step 2. Initialize a dictionary so to store the normalization parameters.
    normalization_map = {}

    # Step 3. Iterate through all the columns.
    for column in normalized_df.columns:
        # Step 4. Check is the type of the column is numeric (float64 or int64) and is not labeled as CLASS or ID.
        if normalized_df[column].dtype in ["float64", "int64"] and column not in ["CLASS", "ID"]:
            # Step 5. Check if the normalization type is min-max.
            if normalizationtype == "minmax":
                # Step 6. Apply min-max normalization.
                minimum_value = normalized_df[column].min()
                maximum_value = normalized_df[column].max()
                normalized_df[column] = (normalized_df[column] - minimum_value) / maximum_value - minimum_value
                # Step 7. Store the normalization parameters.
                normalization_map[column] = (normalizationtype, minimum_value, maximum_value)
            # Step 8. Check if the normalization type is z-normalization / zscore.
            elif normalizationtype == "zscore":
                # Step 9. Apply the z-normalization.
                mean_value = normalized_df[column].mean()
                std_value = normalized_df[column].std()
                normalized_df[column] = (normalized_df[column] - mean_value) / std_value
                #Step 10. Store the normalization parameters.
                normalization_map[column] = (normalizationtype, mean_value, std_value)
            # Step 10. Handle unexpected normalization type
            else:
                raise TypeError("Unsupported normalization type!")
            
    return normalized_df, normalization_map

def apply_normalization(df, normalization):
    
    # Step 1. Copy the input dataframe.
    copy_df = df.copy()

    # Step 2. Iterate through the normalization dictionary.
    for column, values in normalization.items():

        # Step 3. Check if the column exists in the dataframe.
        if column in copy_df.columns:

            # Step 4. Unpack the normalization parameters.
            normalization_type, first_variable, second_variable = values

        # Step 5. Check if the normalization type is min-max
        if normalization_type == "minmax":

            # Step 6. Apply the min-max normalization
            copy_df[column] = (copy_df[column] - first_variable) / (second_variable - first_variable)

            # Step 7. Ensure that the values are in the range of [0, 1]
            # Find documentation here: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
            copy_df[column] = np.clip(copy_df[column], 0, 1)
        # Step 8. Check if the normalization type is z-normalization / zscore.
        elif normalization_type == "zscore":
            # Step 9. Apply the z-normalization
            copy_df[column] = (copy_df[column] - first_variable) / second_variable
        # Step 10. Handle unexpected normalization type
        else:
            raise TypeError("Unsupported normalization type!")

    return copy_df

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

def accuracy(df: pd.DataFrame, correctlabels: list):

    if len(df) != len(correctlabels):
        raise ValueError("The number of rows in the DataFrame must equal the number of correct labels")
    
    # Step 2: Find the label with the highest probability in each row
    predict_labels = df.idxmax(axis=1)
    #print(correctlabels)
    #print(predict_labels)
    # Step 3: Compare with the correct labels
    matches = predict_labels == correctlabels

    # Step 4: Calculate accuracy 
    accuracy = matches.sum() / len(correctlabels)

    return accuracy


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



# Define the class kNN with three functions __init__, fit and predict (after the comments):
#
class kNN:
# Input to __init__: 
# self - the object itself
#
# Output from __init__:
# <nothing>
# 
# This function does not return anything but just initializes the following attributes of the object (self) to None:
# column_filter, imputation, normalization, one_hot, labels, training_labels, training_data, training_time
#
    def __init__(self) -> None:
        self.column_filter = None
        self.imputation = None
        self.normalization = None
        self.one_hot = None
        self.labels = None
        self.training_labels = None
        self.training_data = None
        self.training_time = None

# Input to fit:
# self              - the object itself
# df                - a dataframe (where the column names "CLASS" and "ID" have special meaning)
# normalizationtype - "minmax" (default) or "zscore"
#
# Output from fit:
# <nothing>
#
# The result of applying this function should be:
#
    def fit(self, df: pd.DataFrame, normalizationtype: str = "minmax"):

        # self.column_filter   - a column filter (see Assignment 1) from df
        _,  self.column_filter = create_column_filter(df=df)
        df_filtered = apply_column_filter(df=df, column_filter=self.column_filter)

        # self.imputation      - an imputation mapping (see Assignment 1) from df        
        _, self.imputation = create_imputation(df=df)
        df_imputed = apply_imputation(df=df, imputation=self.imputation)

        # self.normalization   - a normalization mapping (see Assignment 1), using normalizationtype from the imputed df
        _, self.normalization = create_normalization(df=df, normalizationtype=normalizationtype)
        df_normalized = apply_normalization(df=df, normalization=self.normalization)


        if df.select_dtypes(include=['object', 'category']).shape[1] > 0:
            # self.one_hot         - a one-hot mapping (see Assignment 1)
            _, self.one_hot = create_one_hot(df=df)
            df_final = apply_one_hot(df=df, one_hot=self.one_hot)
        else:
            df_final = df_normalized

        # self.training_labels - a pandas series corresponding to the "CLASS" column, set to be of type "category" 
        self.training_labels = df_final['CLASS'].astype('category')
        # self.labels          - a list of the categories (class labels) of the previous series
        self.labels = self.training_labels.cat.categories

        # self.training_data   - the values (an ndarray) of the transformed dataframe, i.e., after employing imputation, 
        # normalization, and possibly one-hot encoding, and also after removing the "CLASS" and "ID" columns
        self.training_data = df_final.drop(columns=['CLASS', 'ID']).values


# Note that the function does not return anything but just assigns values to the attributes of the object.
#
# Input to predict:
# self - the object itself
# df   - a dataframe
# k    - an integer >= 1 (default = 5)
# 
# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are estimated by the relative class frequencies in the set of class labels from the k nearest 
#               (with respect to Euclidean distance) neighbors in training_data
#
# Hint 1: Drop any "CLASS" and "ID" columns first and then apply column filtering, imputation, normalization and one-hot
#
# Hint 2: Get the numerical values (as an ndarray) from the resulting dataframe and iterate over the rows 
#         calling some sub-function, e.g., get_nearest_neighbor_predictions(x_test,k), which for a test row
#         (numerical input feature values) finds the k nearest neighbors and calculate the class probabilities.
#
# Hint 3: This sub-function may first find the distances to all training instances, e.g., pairs consisting of
#         training instance index and distance, and then sort them according to distance, and then (using the indexes
#         of the k closest instances) find the corresponding labels and calculate the relative class frequencies
    def predict(self, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:

   # Preprocess the test data
        # Hint 1: Apply the preprocessing steps (column filtering, imputation, normalization, one-hot encoding)
        df_processed = self._preprocess_data(df=df)

        # Convert processed DataFrame to ndarray for distance computation
        test_data = df_processed.values

        # Initialize an empty DataFrame for predictions
        predictions = []

        # Iterate over each test instance and get predictions
        # Hint 2 & 3: Use a sub-function for finding k nearest neighbors and calculating class probabilities
        for test_instance in test_data:
            class_probabilities = self._get_nearest_neighbor_predictions(test_instance=test_instance, k=k)
            predictions.append(class_probabilities)


        predictions_df = pd.concat(predictions, axis=1).T
        predictions_df.columns = self.labels
        predictions_df.index = df.index

        return predictions_df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data by applying column filter, imputation, normalization, and one-hot encoding"""
        df_filtered = apply_column_filter(df, self.column_filter)
        df_imputed = apply_imputation(df_filtered, self.imputation)
        df_normalized = apply_normalization(df_imputed, self.normalization)
        if self.one_hot:
            df_final = apply_one_hot(df_normalized, self.one_hot)
        else:
            df_final = df_normalized

        # Drop 'CLASS' and 'ID' columns if present
        return df_final.drop(columns=['CLASS', 'ID'], errors='ignore')

    def _get_nearest_neighbor_predictions(self, test_instance: np.ndarray, k: int) -> pd.Series:
        """Finds the k nearest neighbors and calculates class probabilities"""
        # Calculate all distances between test_instance and training_data
        distances = distance.cdist([test_instance], self.training_data, 'euclidean').flatten()

        # Find indices of k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Get labels of nearest neighbors
        nearest_labels = self.training_labels.iloc[nearest_indices]

        # Calculate and return class probabilities
        class_probabilities = nearest_labels.value_counts(normalize=True).reindex(self.labels, fill_value=0)
        return class_probabilities




glass_train_df = pd.read_csv("../glass_train.csv")

glass_test_df = pd.read_csv("../glass_test.csv")

knn_model = kNN()

t0 = time.perf_counter()
knn_model.fit(glass_train_df)
print("Training time: {0:.2f} s.".format(time.perf_counter()-t0))

test_labels = glass_test_df["CLASS"]

k_values = [1,3,5,7,9]
results = np.empty((len(k_values),3))

for i in range(len(k_values)):
    t0 = time.perf_counter()
    predictions = knn_model.predict(glass_test_df,k=k_values[i])
    print("Testing time (k={0}): {1:.2f} s.".format(k_values[i],time.perf_counter()-t0))
    results[i] = [accuracy(predictions,test_labels),brier_score(predictions,test_labels),
                  auc(predictions,test_labels)] # Assuming that you have defined auc - remove otherwise
    

results = pd.DataFrame(results,index=k_values,columns=["Accuracy","Brier score","AUC"])

print()
print("results",results)