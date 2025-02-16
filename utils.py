import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import torch_geometric.datasets
import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import plotly.graph_objects as go
import lmdb


atom_colors = {
    'H': '#FFFFFF',  # Hydrogen - White
    'C': '#2C2C2C',  # Carbon - Dark Gray
    'N': '#3050F8',  # Nitrogen - Deep Blue
    'O': '#FF0D0D',  # Oxygen - Bright Red
    'F': '#90E050'   # Fluorine - Light Green
}


def project_3d_to_2d(pos_3d):
    """Project 3D coordinates to 2D using PCA-like projection"""
    pos_3d = pos_3d.cpu().numpy()
    pos_3d = pos_3d - pos_3d.mean(axis=0)
    _, _, vh = np.linalg.svd(pos_3d)
    return pos_3d @ vh.T[:, :2]


# Load the QM9 dataset
@st.cache_data
def load_dataset():
    return torch_geometric.datasets.QM9(root='data/QM9')


@st.cache_data
def find_molecule(name_or_smiles, _dataset):
    """Cache molecule search results"""
    for data in _dataset:
        if data.name == name_or_smiles or data.smiles == name_or_smiles:
            mol = Chem.MolFromSmiles(data.smiles, sanitize=True)
            return data, mol
    return None, None


def get_atom_symbol(features):
    """Convert QM9 one-hot encoded features to atom symbol"""
    atom_types = ['H', 'C', 'N', 'O', 'F']
    idx = features[:5].argmax().item()
    return atom_types[idx]


@st.cache_data
def get_data_ranges(db_path, graph_name):
    """Get the valid ranges for epoch and ensemble for a given graph"""
    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        epochs = set()
        ensembles = set()

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key_str = key.decode()
                if graph_name in key_str:
                    # Parse key components
                    parts = key_str.split('-')
                    if len(parts) >= 4:
                        epochs.add(int(parts[0]))
                        ensembles.add(int(parts[2]))

        env.close()
        return {
            'epoch_range': (min(epochs), max(epochs)) if epochs else (0, 0),
            'ensemble_range': (min(ensembles), max(ensembles)) if ensembles else (0, 0)
        }
    except lmdb.Error as e:
        st.error(f"Error opening LMDB database: {str(e)}")
        return {'epoch_range': (0, 0), 'ensemble_range': (0, 0)}