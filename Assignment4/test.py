# Import necessary libraries
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as descriptors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Function to extract features from a SMILES string
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_weight = descriptors.CalcExactMolWt(mol)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return mol_weight, morgan_fp

# Function to prepare data for model training
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['MolWeight'], df['MorganFP'] = zip(*df['SMILES'].apply(extract_features))
    X = np.array(list(df['MorganFP']))
    return df, X

def train_and_evaluate(X, y, model, n_splits=5, use_kfold=False):
    fig, ax = plt.subplots()

    if use_kfold:
        # Use k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_splits)
        auc_scores = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred)
            auc_scores.append(auc_score)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            ax.plot(fpr, tpr, alpha=0.3, label=f'Fold {i+1} (AUC = {auc_score:.2f})')

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        ax.plot(mean_fpr, mean_tpr, color='blue',
                label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                lw=2, alpha=0.8)

    else:
        # Use simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        ax.plot(fpr, tpr, color='blue', label=f'Simple Split (AUC = {auc_score:.2f})', lw=2, alpha=0.8)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', alpha=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    plt.show()


    return auc_score if not use_kfold else mean_auc


# Load and prepare the training data
train_data_df, X_train = prepare_data("training_smiles.csv")
y_train = train_data_df['ACTIVE'].values

# User input for model selection
model_choice = input("Select a model: 1 for Random Forest, 2 for Support Vector Machine: ")
if model_choice == '1':
    model = RandomForestClassifier()
elif model_choice == '2':
    model = SVC(probability=True)

# User input for cross-validation choice
cv_choice = input("Select validation method: 1 for simple split, 2 for k-fold cross-validation: ")
if cv_choice == '2':
    k = int(input("Enter the number of folds for k-fold cross-validation: "))
    use_kfold = True
else:
    k = 0
    use_kfold = False

# Train and evaluate the selected model
auc_score = train_and_evaluate(X_train, y_train, model, n_splits=k, use_kfold=use_kfold)
print(f"AUC score: {auc_score}")