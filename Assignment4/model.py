from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from app import *

# Models to test and their corresponding AUC scores
models_and_auc_scores = {"Random Forest": [RandomForestClassifier(), 0], "Naive Bayes": [GaussianNB(), 0], "Logistic Regression": [LogisticRegression(), 0], "KNN": [KNeighborsClassifier(), 0], "Gradient Boosting": [GradientBoostingClassifier(), 0], "Decision Tree": [DecisionTreeClassifier(), 0], "Extreme Random Forest": [ExtraTreesClassifier(), 0]}

# Divide the data into training and validation sets, where 'ACTIVE' is the target variable
def split(df):
    print("Splitting data...")
    X = df.drop('ACTIVE', axis=1)
    y = df['ACTIVE']
    # validation set size is 20% of the total data, training set size is 80%
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_valid, y_train, y_valid

# Feature preprocessing pipeline (transfom data)
def transform_pipeline(X_train):
    print("Transforming data...")

    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=['float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns

    # Create a pipeline for numerical columns which imputes missing values with the median and applies PCA
    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('pca', PCA(n_components=0.95)),
        ])

    # Create a pipeline for categorical columns which imputes missing values with 'missing' and encodes the values
    categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

    # Create a preprocessor which applies the numerical and categorical pipelines to the correct columns
    preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_and_evaluate_models(X_train, X_valid, y_train, y_valid, preprocessor):
    print("Training and evaluating models...")

    # Loop through each model in models_and_auc_scores and train it
    for model_name, (model, _) in models_and_auc_scores.items():
        print(f"Making pipeline for {model_name}...")
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            (model_name, model)
        ])
        # Fit each model
        print(f"Fitting model for {model_name}...")
        model_pipeline.fit(X_train, y_train)

        # Make predictions on the validation set
        try:
            print(f"Predicting for {model_name}...")
            y_pred_prob = model_pipeline.predict_proba(X_valid)[:, 1]
        except ValueError as e:
            print(f"Error for model {model_name}: {e}")
            continue

        # Calculate AUC score for the model
        auc = roc_auc_score(y_valid, y_pred_prob)

        # Update the AUC score for the model in the dictionary
        models_and_auc_scores[model_name][1] = auc

        print ("Model:", model_name, "- AUC score:", auc)
    print("Best model:", max(models_and_auc_scores, key=lambda x: models_and_auc_scores[x][1]))
    print("AUC scores:", models_and_auc_scores)

def tune_best_model(Xy_train, preprocessor):
    print("Tuning best model...")
    X_train, y_train = Xy_train
    best_model = max(models_and_auc_scores, key=lambda x: models_and_auc_scores[x][1])
    return None

def test_best_model(Xy_test, preprocessor):
    print("Testing best model...")
    X_test, y_test = Xy_test

    return None

def predict_activity(model, Xy_test):
    print("Predicting activity...")
    return None

def main():
    df = load_data("training_smiles.csv")
    mole = extract_mole(df)
    #all_features = extract_features(df)
    final_features = extract_top_features(df)
    clean_df = clean_data(final_features)

    # Train and evaluate models
    X_train, X_valid, y_train, y_valid = split(clean_df)
    preprocessor = transform_pipeline(X_train)
    train_and_evaluate_models(X_train, X_valid, y_train, y_valid, preprocessor)

    #DONE: 
    # Split data
    # Transform data
    # Train models
    # Evaluate models

    #TODO: 
    # fix imbalanced dataset
    # Tune model hyperparameters
    # Predict with best model on test set

if __name__ == "__main__":
    main()

