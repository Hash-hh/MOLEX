import streamlit as st
from rdkit import Chem
import torch_geometric.datasets
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import lmdb
import json

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
    """Get available epochs and ensembles for a specific graph"""
    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        epochs = set()
        ensembles = set()

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key_parts = key.decode().split('-')
                if len(key_parts) >= 4 and key_parts[3] == graph_name:
                    epochs.add(int(key_parts[0]))
                    ensembles.add(int(key_parts[2]))

        env.close()

        if not epochs or not ensembles:
            return {
                'epoch_values': [0],
                'epoch_range': (0, 0),
                'epoch_step': 1,
                'ensemble_values': [0],
                'ensemble_range': (0, 0)
            }

        sorted_epochs = sorted(list(epochs))
        sorted_ensembles = sorted(list(ensembles))

        # Calculate the step size if there are at least 2 epochs
        epoch_step = 1
        if len(sorted_epochs) >= 2:
            # Find the difference between consecutive epochs
            diffs = [sorted_epochs[i + 1] - sorted_epochs[i] for i in range(len(sorted_epochs) - 1)]
            # Use the most common difference as the step
            if diffs:
                from collections import Counter
                epoch_step = Counter(diffs).most_common(1)[0][0]

        return {
            'epoch_values': sorted_epochs,
            'epoch_range': (min(epochs), max(epochs)),
            'epoch_step': epoch_step,
            'ensemble_values': sorted_ensembles,
            'ensemble_range': (min(ensembles), max(ensembles))
        }
    except lmdb.Error as e:
        return {
            'epoch_values': [0],
            'epoch_range': (0, 0),
            'epoch_step': 1,
            'ensemble_values': [0],
            'ensemble_range': (0, 0)
        }

def aggregate_edge_data(db_path, molecule_name, epoch_list, ensemble_range, status):
    """
    Aggregate edge data across specified epochs and ensembles.
    For add edges: count edges with prob > 0 (these are edges we want to add)
    For delete edges: count edges with prob = 0 (these are edges we want to delete)
    """
    add_counts = {}
    del_counts = {}

    try:
        env = lmdb.open(db_path, readonly=True, lock=False)

        with env.begin() as txn:
            cursor = txn.cursor()
            for epoch in epoch_list:
                for ensemble in range(ensemble_range[0], ensemble_range[1] + 1):
                    key = f'{epoch}-{status}-{ensemble}-{molecule_name}'
                    value = txn.get(key.encode())

                    if value:
                        data = json.loads(value.decode())

                        # Process add edges
                        add_edges = list(zip(data['rewire_add_edge_index'][0], data['rewire_add_edge_index'][1]))
                        for i, edge in enumerate(add_edges):
                            if edge not in add_counts:
                                add_counts[edge] = 0
                            add_counts[edge] += 1

                        # Process delete edges
                        del_edges = list(zip(data['rewire_del_edge_index'][0], data['rewire_del_edge_index'][1]))
                        for i, edge in enumerate(del_edges):
                            if edge not in del_counts:
                                del_counts[edge] = 0
                            del_counts[edge] += 1

        env.close()
        return add_counts, del_counts
    except lmdb.Error as e:
        return {}, {}


def create_edge_histograms(add_counts, del_counts, total_samples):
    """Create histograms for add and delete edge frequencies"""
    # Convert to DataFrames with proper edge formatting
    add_df = pd.DataFrame([
        {"edge": f"({edge[0]}, {edge[1]})",
         "count": count,
         "frequency": count / total_samples}
        for edge, count in add_counts.items()
    ])
    del_df = pd.DataFrame([
        {"edge": f"({edge[0]}, {edge[1]})",
         "count": count,
         "frequency": count / total_samples}
        for edge, count in del_counts.items()
    ])

    if not add_df.empty:
        # Sort by frequency
        add_df = add_df.sort_values("frequency", ascending=False)

        # Create add edges histogram
        fig_add = px.bar(
            add_df,
            x="edge",
            y="frequency",
            title="Add Edge Frequencies",
            labels={
                "edge": "Edge Pairs (source, target)",
                "frequency": "Selection Frequency"
            },
            color="frequency",
            color_continuous_scale="Greens"
        )

        # Update layout
        fig_add.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=400,
            title_x=0.5,
            yaxis_title="Frequency",
            xaxis_title="Edge Pairs (source, target)",
            xaxis=dict(
                type='category',  # Force categorical x-axis
                tickmode='array',
                ticktext=add_df['edge'],
                tickvals=list(range(len(add_df)))
            )
        )
    else:
        fig_add = go.Figure()
        fig_add.update_layout(
            title="No Add Edges Found",
            title_x=0.5
        )

    if not del_df.empty:
        # Sort by frequency
        del_df = del_df.sort_values("frequency", ascending=False)

        # Create delete edges histogram
        fig_del = px.bar(
            del_df,
            x="edge",
            y="frequency",
            title="Delete Edge Frequencies",
            labels={
                "edge": "Edge Pairs (source, target)",
                "frequency": "Selection Frequency"
            },
            color="frequency",
            color_continuous_scale="Reds"
        )

        # Update layout
        fig_del.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=400,
            title_x=0.5,
            yaxis_title="Frequency",
            xaxis_title="Edge Pairs (source, target)",
            xaxis=dict(
                type='category',  # Force categorical x-axis
                tickmode='array',
                ticktext=del_df['edge'],
                tickvals=list(range(len(del_df)))
            )
        )
    else:
        fig_del = go.Figure()
        fig_del.update_layout(
            title="No Delete Edges Found",
            title_x=0.5
        )

    return fig_add, fig_del
