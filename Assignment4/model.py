from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Models to test and their corresponding AUC scores
models_and_auc_scores = {"Random Forest": [RandomForestClassifier(), 0], "Naive Bayes": [GaussianNB(), 0], "Logistic Regression": [LogisticRegression(), 0], "SVM": [SVC(), 0], "KNN": [KNeighborsClassifier(), 0], "Gradient Boosting": [GradientBoostingClassifier(), 0], "Decision Tree": [DecisionTreeClassifier(), 0]}

# Divide the data into training and validation sets, where 'ACTIVE' is the target variable
def split(df):
    X = df.drop('ACTIVE', axis=1)
    y = df['ACTIVE']
    # validation set size is 20% of the total data, training set size is 80%
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

# Feature preprocessing pipeline (transfom data)
def transform_pipeline(Xy_train):
    X_train, _ = Xy_train

    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=['float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Create a pipeline for numerical columns which imputes missing values with the median and scales the data
    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    # Create a pipeline for categorical columns which imputes missing values with 'missing'
    categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])

    # Create a preprocessor which applies the numerical and categorical pipelines to the correct columns
    preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def train_and_evaluate_models(Xy_train, Xy_valid, preprocessor):
    X_train, y_train = Xy_train
    X_valid, y_valid = Xy_valid
    # Loop through each model in models_and_auc_scores and train it
    for model_name, (model, auc_score) in models_and_auc_scores.items():
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor), 
            (model_name, model)
        ])
        # Fit each model
        model_pipeline.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred_prob = model.predict_proba(X_valid)[:, 1]

        # Calculate AUC score for the model
        auc = roc_auc_score(y_valid, y_pred_prob)

        # Update the AUC score for the model in the dictionary
        models_and_auc_scores[model_name][1] = auc

        print ("Model:", model_name, "AUC score:", auc)
    print("Best model:", max(models_and_auc_scores, key=lambda x: models_and_auc_scores[x][1]))
    
def evaluate_models(Xy_train, Xy_valid):
    X_train, y_train = Xy_train
    X_valid, y_valid = Xy_valid

    # iterate through each model in models_and_auc_scores
    for model in models_and_auc_scores:
        # fit the model
        models_and_auc_scores[model][0].fit(X_train, y_train)
        # calculate the AUC score
        models_and_auc_scores[model][1] = roc_auc_score(y_valid, models_and_auc_scores[model][0].predict_proba(X_valid)[:,1])

    # sort the models_and_auc_scores dictionary by the AUC score
    sorted_models_and_auc_scores = sorted(models_and_auc_scores.items(), key=lambda x: x[1][1], reverse=True)
    # print the sorted dictionary
    print(sorted_models_and_auc_scores)

def main():
    # Split data
    # Transform data
    # Train models
    # Evaluate models
    # Predict with best model on test set

    #TODO: fix imbalance dataset
    return 0

if __name__ == "__main__":
    main()

