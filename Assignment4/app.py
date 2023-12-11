import rdkit
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sn
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Fragments
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

from sklearn.feature_selection import SelectKBest, chi2, RFECV, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

name = 'Assignment4/training_smiles.csv'

def load_data():
    df = pd.read_csv(name, index_col=0)
    return df

def extract_mole(df):
    # replace SMILES column with mole
    df['mole'] = df['SMILES'].apply(rdkit.Chem.MolFromSmiles)
    df.drop('SMILES', axis=1, inplace=True)
    return df

# now we can extract some features from the dataframe
# instead of making a function for each feature we utilize the lambda function

def extract_features(df):
    # 29 features extracted:
    # molecular descriptors:
    df['molecularWeight'] = df['mole'].apply(lambda x: Descriptors.MolWt(x))
    df['exactMolWt'] = df['mole'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['TPSA'] = df['mole'].apply(lambda x: Descriptors.TPSA(x))
    
    # atom counts:
    df['numAtoms'] = df['mole'].apply(lambda x: x.GetNumAtoms())
    df['numHeavyAtoms'] = df['mole'].apply(lambda x: x.GetNumHeavyAtoms())
    df['numHeteroAtoms'] = df['mole'].apply(lambda x: Lipinski.NumHeteroatoms(x))
    df['NHOH_count'] = df['mole'].apply(lambda x: Lipinski.NHOHCount(x))
    df['NO_count'] = df['mole'].apply(lambda x: Lipinski.NOCount(x))

    # fingerprint features:
    df['morganFingerprint'] = df['mole'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))
    ## morgan fingerprint must be represented as a string
    df['morganFingerprint'] = df['morganFingerprint'].apply(lambda x: x.ToBitString())

    # functional group counts:
    df['AI_COO'] = df['mole'].apply(lambda x: Descriptors.fr_Al_COO(x))
    df['Ar_N'] = df['mole'].apply(lambda x: Fragments.fr_Ar_N(x))
    df['COO'] = df['mole'].apply(lambda x: Fragments.fr_COO(x))

    # other descriptors:
    df['numBonds'] = df['mole'].apply(lambda x: x.GetNumBonds())
    df['RotatableBonds'] = df['mole'].apply(lambda x: Descriptors.NumRotatableBonds(x))
    df['HDonors'] = df['mole'].apply(lambda x: Descriptors.NumHDonors(x))
    df['HAcceptors'] = df['mole'].apply(lambda x: Descriptors.NumHAcceptors(x))
    df['numAliphaticRings'] = df['mole'].apply(lambda x: rdMolDescriptors.CalcNumAliphaticRings(x))
    df['numAromaticRings'] = df['mole'].apply(lambda x: rdMolDescriptors.CalcNumAromaticRings(x))
    df['numSaturatedRings'] = df['mole'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    df['numRadicalElectron'] = df['mole'].apply(lambda x: Descriptors.NumRadicalElectrons(x))
    df['numValenceElectrons'] = df['mole'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    df['num_H_acceptors'] = df['mole'].apply(lambda x: Lipinski.NumHAcceptors(x))
    df['num_H_donors'] = df['mole'].apply(lambda x: Lipinski.NumHDonors(x))
    df['numSaturatedRings'] = df['mole'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    df['amideAmount'] = df['mole'].apply(lambda x: Fragments.fr_amide(x))
    df['benzeneAmount'] = df['mole'].apply(lambda x: Fragments.fr_benzene(x))
    df['esterAmount'] = df['mole'].apply(lambda x: Fragments.fr_ester(x))
    df['nitroAmount'] = df['mole'].apply(lambda x: Fragments.fr_nitro(x))
    df['nitroArom'] = df['mole'].apply(lambda x: Fragments.fr_nitro_arom(x))
    
    # now we can drop the mole column
    df.drop('mole', axis=1, inplace=True)
    clean_data(df)
    # df.to_csv('Assignment4/features.csv')

    return df

def feature_selection(df):
    print(pick_top_features(df))

def pick_top_features(df):
    df1 = df.copy()
    y = df1['ACTIVE']
    X = df1.drop('ACTIVE', axis=1)

    # get top 10 features by calculating correlation
    top_features = SelectKBest(score_func=chi2, k=10)
    fit = top_features.fit(X,y)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)
    featureScores = pd.concat([columns,scores],axis=1)
    featureScores.columns = ['Name','Score']
    
    # print features with the top 10 scores
    return featureScores.nlargest(10, 'Score')

# def rfe_selection(df):
#     df2 = extract_features(df)
#     df1 = df2.copy()
#     y = df1['ACTIVE']
#     X = df1.drop('ACTIVE', axis=1)

#     clf = RandomForestClassifier()

#     # Initialize RFE with the classifier and the desired number of features to select (e.g., 10)
#     rfe = RFE(clf, n_features_to_select=10)

#     # Fit RFE on the training data
#     rfe.fit(X, y)

#     # Get the selected features
#     selected_features = X.columns[rfe.support_]

#     # Train a model using only the selected features
#     clf.fit(X[selected_features], y)

#     # Make predictions on the test set
#     y_pred = clf.predict(X[selected_features])

#     # Evaluate the accuracy of the model
#     accuracy = accuracy_score(y, y_pred)
#     print(f"Selected Features: {selected_features}")
#     print(f"Accuracy on Test Set: {accuracy}")

def extract_top_features(df):
    # RESULT FROM pick_top_features:
    # 1. morganFingerprint
    df['morganFingerprint'] = df['mole'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))
    ## morgan fingerprint must be represented as a string
    df['morganFingerprint'] = df['morganFingerprint'].apply(lambda x: x.ToBitString())
    # 2. molecularWeight
    df['molecularWeight'] = df['mole'].apply(lambda x: Descriptors.MolWt(x))
    # 3. exactMolWt
    df['exactMolWt'] = df['mole'].apply(lambda x: Descriptors.ExactMolWt(x))
    # 4. numValenceElectrons
    df['numValenceElectrons'] = df['mole'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    # 5. amideAmount
    df['amideAmount'] = df['mole'].apply(lambda x: Fragments.fr_amide(x))
    # 6. TPSA
    df['TPSA'] = df['mole'].apply(lambda x: Descriptors.TPSA(x))
    # 7. numBonds
    df['numBonds'] = df['mole'].apply(lambda x: x.GetNumBonds())
    # 8. numAromaticRings
    df['numAromaticRings'] = df['mole'].apply(lambda x: rdMolDescriptors.CalcNumAromaticRings(x))
    # 9. Ar_N
    df['Ar_N'] = df['mole'].apply(lambda x: Fragments.fr_Ar_N(x))
    # 10. numHeavyAtoms
    df['numHeavyAtoms'] = df['mole'].apply(lambda x: x.GetNumHeavyAtoms())
        
    df.drop('mole', axis=1, inplace=True)
    clean_data(df)
    # df.to_csv('Assignment4/features.csv')

    return df


def clean_data(df):
    last = df.pop('ACTIVE')
    df['ACTIVE'] = last
    df.drop_duplicates()
    df.isnull()
    return df

def data_analysis(df):
    df1 = df.copy()
    # find correlation
    corr_matrix = df1.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()

    # distribution for active and inactive 
    numY, numN = df.ACTIVE.value_counts()
    print(numY, numN)
    df.ACTIVE.value_counts().plot(kind='pie',autopct='%1.0f%%', colors=['royalblue','red'])
    plt.xlabel("ACTIVE")
    plt.ylabel("INACTIVE")
    plt.show()

df = load_data()
extract_mole(df)
# extract_features(df)
extract_top_features(df) #returns the top 10 features
# print(extract_top_features(df))
# rfe_selection(df)
# data_analysis(df)
# feature_selection(df)
