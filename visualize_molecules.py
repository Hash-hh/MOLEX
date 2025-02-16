from rdkit import Chem
from rdkit.Chem import Draw
import torch_geometric.datasets
import tqdm

# Load the QM9 dataset
dataset = torch_geometric.datasets.QM9(root='data/QM9')

# Define a function to get molecule from dataset by name or SMILES
def get_molecule(name_or_smiles):
    # check if user entered a name or SMILES
    if 'gdb_' in name_or_smiles:
        name = name_or_smiles
        smiles = None
    else:
        name = None
        smiles = name_or_smiles

    for i, data in enumerate(tqdm.tqdm(dataset, desc='Searching')):
        if data.name == name or data.smiles == smiles:
            mol = Chem.MolFromSmiles(data.smiles)
            if mol:
                return mol

# Input for molecule name or SMILES
name_or_smiles = 'gdb_13394'

if name_or_smiles:
    mol = get_molecule(name_or_smiles)
    if mol:
        print('Molecule found!')
        img = Draw.MolToImage(mol)
        img.save('mol.png')
    else:
        print('Molecule not found in the QM9 dataset.')
