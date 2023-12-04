from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
import numpy as np


m = Chem.MolFromSmiles('Cc1ccccc1')
fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)

print(m.GetNumAtoms())
print(d.CalcExactMolWt(m))
print(f.fr_Al_COO(m))
print(l.HeavyAtomCount(m))
print(np.array(fp))