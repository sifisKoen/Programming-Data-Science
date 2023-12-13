from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from app import *
from imblearn.over_sampling import SMOTE

# Models to test and their corresponding AUC scores
models_and_auc_scores = {"Random Forest": [RandomForestClassifier(), 0], "Naive Bayes": [GaussianNB(), 0], "Logistic Regression": [LogisticRegression(), 0], "KNN": [KNeighborsClassifier(), 0], "Gradient Boosting": [GradientBoostingClassifier(), 0], "Decision Tree": [DecisionTreeClassifier(), 0], "Extreme Random Forest": [ExtraTreesClassifier(), 0]}

# Divide the data into training and validation sets, where 'ACTIVE' is the target variable
def split(df):
    print("Splitting data...")
    X = df.drop('ACTIVE', axis=1)
    y = df['ACTIVE']
    # validation set size is 20% of the total data, training set size is 80%
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
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
    best_model = None
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

        # Update best model variable if the current model is better than the current best model
        if best_model == None or auc > models_and_auc_scores[best_model][1]:
            best_model = model_name

    print("Best model:", max(models_and_auc_scores, key=lambda x: models_and_auc_scores[x][1]))
    print("AUC scores:", models_and_auc_scores,"\n")
    
    # Return best model
    return best_model, models_and_auc_scores[best_model][0]

def tune_best_model(X_valid, y_valid, preprocessor, best_model, best_model_name):
    print("Tuning best model...")

    # Parameters to tune for the best model (logistic regression) to find the best model
    logistic_regression_param_grid = {
        'best_model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'best_model__solver': ['lbfgs', 'liblinear'],
        'best_model__max_iter': [100,200,300,400,500,600,700,800,900,1000]
    }

    # Parameters to tune for the best model (gradient boosting) to find the best model
    gradient_boosting_param_grid = {
        'best_model__n_estimators': [100, 200, 300],
        'best_model__learning_rate': [0.001, 0.01, 0.1],
        'best_model__max_depth': [3, 5, 7],
        'best_model__min_samples_split': [2, 5, 10],
        'best_model__min_samples_leaf': [1, 2, 4],
        'best_model__subsample': [0.8, 0.9, 1.0],
        'best_model__max_features': ["auto", "sqrt", "log2", None],
    }

    # Parameters to tune for the best model (naive bayes) to find the best model
    naive_bayes_param_grid = {
        'best_model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    }

    # Dictionary of model names and their corresponding parameter grids
    model_param_dict = {"Logistic Regression": logistic_regression_param_grid, "Gradient Boosting": gradient_boosting_param_grid, "Naive Bayes": naive_bayes_param_grid}

    # Create a pipeline with the preprocessor, scaling of data and the best model
    tuning_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('best_model', best_model)
    ])

    # Perform grid search to find best hyperparameters for the best model
    print("Performing grid search...")
    grid_search = GridSearchCV(tuning_pipeline, model_param_dict[best_model_name], cv=5, scoring='roc_auc', verbose=3, n_jobs=-1)
    grid_search.fit(X_valid, y_valid)

    best_parameters = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    print("Best parameters:", best_parameters)
    print("Best estimator:", grid_search.best_estimator_)
    print("Best score:", grid_search.best_score_)

    return best_parameters, best_estimator

def test_best_model(X_test, y_test, preprocessor, best_parameters, best_estimator):
    threshold = 0.5
    print("Testing best model...")
    best_model = best_estimator
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC-ROC Score: {auc}')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)
    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall}')
    f1 = f1_score(y_test, y_pred)
    print(f'F1 Score: {f1}')

# def test_best_model(X_test, y_test, preprocessor, best_parameters, best_estimator, num_runs=10):
#     print("Testing best model...")

#     best_model = best_estimator
#     auc_scores = []
#     accuracy_scores = []
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []

#     for run in range(num_runs):
#         print(f"Run {run + 1}/{num_runs}")

#         # Shuffle the data for each run
#         X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=run)

#         print("making pipeline...")
#         # Modify the pipeline if necessary
#         test_pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('scaler', StandardScaler()),
#             ('best_model', GaussianNB().set_params(**best_parameters[1]))
#         ])

#         print("evaluating model...")
#         # Evaluate the model
#         print("fetching auc score...")
#         auc = np.mean(cross_val_score(test_pipeline, X_test_shuffled, y_test_shuffled, cv=StratifiedKFold(n_splits=5, shuffle=True),
#                                     scoring='roc_auc', n_jobs=-1))
#         print("fetching accuracy score...")
#         accuracy = np.mean(cross_val_score(test_pipeline, X_test_shuffled, y_test_shuffled, cv=StratifiedKFold(n_splits=5, shuffle=True),
#                                         scoring='accuracy', n_jobs=-1))
#         print("fetching precision score...")
#         precision = np.mean(cross_val_score(test_pipeline, X_test_shuffled, y_test_shuffled, cv=StratifiedKFold(n_splits=5, shuffle=True),
#                                         scoring='precision', n_jobs=-1))
#         print("fetching recall score...")
#         recall = np.mean(cross_val_score(test_pipeline, X_test_shuffled, y_test_shuffled, cv=StratifiedKFold(n_splits=5, shuffle=True),
#                                         scoring='recall', n_jobs=-1))
#         print("fetching f1 score...")
#         f1 = np.mean(cross_val_score(test_pipeline, X_test_shuffled, y_test_shuffled, cv=StratifiedKFold(n_splits=5, shuffle=True),
#                                     scoring='f1', n_jobs=-1))

#         auc_scores.append(auc)
#         accuracy_scores.append(accuracy)
#         precision_scores.append(precision)
#         recall_scores.append(recall)
#         f1_scores.append(f1)

#     # Report average scores
#     print(f"Avg AUC Score: {np.mean(auc_scores)}")
#     print(f"Avg Accuracy: {np.mean(accuracy_scores)}")
#     print(f"Avg Precision: {np.mean(precision_scores)}")
#     print(f"Avg Recall: {np.mean(recall_scores)}")
#     print(f"Avg F1 Score: {np.mean(f1_scores)}")

def predict_activity(model, Xy_test):
    print("Predicting activity...")

    return None

def main():
    df = load_data("training_smiles.csv")
    mole = extract_mole(df)
    #all_features = extract_features(df)
    final_features = extract_top_features(df)

    # Split data into training set for training model to evaluate and testing model after fine tuning (holdout set)
    X_train, X_test, y_train, y_test = split(final_features)

    # Further split the training set into new training set for training model and validation set for evaluating model and fine tuning
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("X_train_final shape:", X_train_final.shape, "\nX_valid shape:", X_valid.shape, "\ny_train_final shape:", y_train_final.shape, "\ny_valid shape:", y_valid.shape, "\nX_test shape:", X_test.shape, "\ny_test shape:", y_test.shape, "\n")

    preprocessor = transform_pipeline(X_train_final)
    # Train and evaluate models with training data
    best_model_name, best_baseline_model = train_and_evaluate_models(X_train_final, X_valid, y_train_final, y_valid, preprocessor)

    # Tune best model with validation data
    best_parameters, best_estimator = tune_best_model(X_valid, y_valid, preprocessor, best_baseline_model, best_model_name)

    # Test best model with optimal hyperparameters with test data
    test_best_model(X_test, y_test, preprocessor, best_parameters, best_estimator)

    #DONE: 
    # Split data
    # Transform data
    # Train models
    # Evaluate models
    # Tune model hyperparameters

    #TODO: 
    # fix imbalanced dataset
    # Predict with best model on test set

if __name__ == "__main__":
    main()

