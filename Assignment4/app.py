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
    # molecular descriptors:
    df['molecularWeight'] = df['mole'].apply(lambda x: Descriptors.MolWt(x))
    df['exactMolWt'] = df['mole'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['TPSA'] = df['mole'].apply(lambda x: Descriptors.TPSA(x))

    # atom counts:
    df['numAtoms'] = df['mole'].apply(lambda x: x.GetNumAtoms())
    df['numHeavyAtoms'] = df['mole'].apply(lambda x: x.GetNumHeavyAtoms())

    # fingerprint features:
    df['morganFingerprint'] = df['mole'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))
    ## morgan fingerprint must be represented as a string
    df['morganFingerprint'] = df['morganFingerprint'].apply(lambda x: x.ToBitString())

    # functional group counts:
    df['AI_COO'] = df['mole'].apply(lambda x: Descriptors.fr_Al_COO(x))

    # other descriptors:
    df['RotatableBonds'] = df['mole'].apply(lambda x: Descriptors.NumRotatableBonds(x))
    df['HDonors'] = df['mole'].apply(lambda x: Descriptors.NumHDonors(x))
    df['HAcceptors'] = df['mole'].apply(lambda x: Descriptors.NumHAcceptors(x))

    # now we can drop the mole column
    df.drop('mole', axis=1, inplace=True)
    clean_data(df)
    # df.to_csv('Assignment4/features.csv')

    return df

# MISSING: how to pick best features

df = load_data()

def clean_data(df):
    last = df.pop('ACTIVE')
    df['ACTIVE'] = last
    df.drop_duplicates()
    df.isnull()
    return df

#change this
def data_analysis(df):
    # check correlation among features
    df_encoded = df.copy()
    #df_encoded.drop('INDEX', axis=1, inplace=True)
    corr_matrix = df_encoded.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()
    # data are highly correlated, we may want to drop number of heavy atoms

    # check distribution of the label
    numY, numN = df.ACTIVE.value_counts()
    print(numY, numN)
    df.ACTIVE.value_counts().plot(kind='pie',autopct='%1.0f%%', colors=['royalblue','red'])
    plt.xlabel("ACTIVE")
    plt.ylabel("INACTIVE")
    plt.show()

extract_mole(df)
extract_features(df)
data_analysis(df)
