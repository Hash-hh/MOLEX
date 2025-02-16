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
import json
import os
from utils import project_3d_to_2d, load_dataset, find_molecule, get_atom_symbol, atom_colors, get_data_ranges


# Set page config to narrow layout
st.set_page_config(layout="centered", page_title="MOLEX Playground", page_icon="🧪")


def create_3d_plotly(pos_3d, atom_labels, edges, rewire_add_edges, rewire_del_edges):
    """Create an interactive 3D plot using Plotly"""
    pos_3d_np = pos_3d.cpu().numpy()

    # Create figure
    fig = go.Figure()

    # Add nodes (atoms)
    fig.add_trace(go.Scatter3d(
        x=pos_3d_np[:, 0],
        y=pos_3d_np[:, 1],
        z=pos_3d_np[:, 2],
        mode='markers+text',
        marker=dict(
            size=15,
            color=[atom_colors.get(atom_labels[i].split(':')[1], 'lightblue') for i in range(len(atom_labels))],
            line_width=2,
            line_color='black'
        ),
        text=[atom_labels[i] for i in range(len(atom_labels))],
        hoverinfo='text',
        textposition="top center",
    ))

    # Add original edges
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[pos_3d_np[edge[0], 0], pos_3d_np[edge[1], 0]],
            y=[pos_3d_np[edge[0], 1], pos_3d_np[edge[1], 1]],
            z=[pos_3d_np[edge[0], 2], pos_3d_np[edge[1], 2]],
            mode='lines',
            line=dict(color='black', width=1),
            hoverinfo='none'
        ))

    # Add rewire add edges
    for edge in rewire_add_edges:
        fig.add_trace(go.Scatter3d(
            x=[pos_3d_np[edge[0], 0], pos_3d_np[edge[1], 0]],
            y=[pos_3d_np[edge[0], 1], pos_3d_np[edge[1], 1]],
            z=[pos_3d_np[edge[0], 2], pos_3d_np[edge[1], 2]],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            hoverinfo='none'
        ))

    # Add rewire delete edges
    for edge in rewire_del_edges:
        fig.add_trace(go.Scatter3d(
            x=[pos_3d_np[edge[0], 0], pos_3d_np[edge[1], 0]],
            y=[pos_3d_np[edge[0], 1], pos_3d_np[edge[1], 1]],
            z=[pos_3d_np[edge[0], 2], pos_3d_np[edge[1], 2]],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            hoverinfo='none'
        ))

    # Update layout
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    return fig


def create_2d_plotly(pos, atom_labels, existing_edges, rewire_add_edges, rewire_del_edges, original_node_count):
    """Create an interactive 2D plot using Plotly"""
    edge_traces = []

    # Add existing edges
    for edge in existing_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color='black'),
            hoverinfo='none',
            mode='lines'
        ))

    # Add rewire add edges
    for edge in rewire_add_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color='green', dash='dash'),
            hoverinfo='none',
            mode='lines'
        ))

    # Add rewire delete edges
    for edge in rewire_del_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color='red', dash='dash'),
            hoverinfo='none',
            mode='lines'
        ))

    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in range(original_node_count):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(atom_labels[node])
        color_name = atom_labels[node].split(':')[1]
        node_colors.append(atom_colors.get(color_name, 'lightblue'))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=20,
            line_width=1,
            line_color='black'
        ),
        textfont=dict(size=10)
    )

    # Create figure
    fig = go.Figure(data=[*edge_traces, node_trace])

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        dragmode='pan'
    )

    # Make the plot square
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


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


@st.cache_resource
def load_lmdb_data(db_path, graph_name, epoch, status, ensemble):
    """Load data from LMDB database for a specific graph configuration"""
    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        results = {}

        with env.begin() as txn:
            cursor = txn.cursor()
            # Construct the specific key we're looking for
            target_key = f'{epoch}-{status}-{ensemble}-{graph_name}'

            for key, value in cursor:
                key_str = key.decode()
                if key_str == target_key:  # Exact match instead of 'in'
                    data = json.loads(value.decode())
                    results[key_str] = data
                    break  # We found our exact match, no need to continue

        env.close()
        return results
    except lmdb.Error as e:
        st.error(f"Error opening LMDB database: {str(e)}")
        return None


# Main app
st.title('MOLEX — Playground')

# Default database path
default_db_path = r"C:\Users\hasha\PycharmProjects\ER-GNN\ER-GNN\logs\zpve_rewire\15-02-25_08.31.43 PM\rewiring_logs.lmdb"

# Main molecule input
name_or_smiles = st.text_input('Enter molecule name or SMILES', 'gdb_26883')

# Add status selector (train/val)
status = st.radio(
    "Select Status",
    ["train", "val"],
    horizontal=True
)

# Load dataset and find molecule
dataset = load_dataset()

if name_or_smiles:
    data, rdkit_mol = find_molecule(name_or_smiles, dataset)

    if data:
        # Get valid ranges for this molecule
        ranges = get_data_ranges(default_db_path, data.name)
        epoch_count = st.slider(
            "Epoch",
            min_value=ranges['epoch_range'][0],
            max_value=ranges['epoch_range'][1],
            value=ranges['epoch_range'][0],
            step=1
        )
        ensemble = st.slider(
            "Ensemble",
            min_value=ranges['ensemble_range'][0],
            max_value=ranges['ensemble_range'][1],
            value=ranges['ensemble_range'][0],
            step=1
            )

    if data and rdkit_mol:
        st.write(f'SMILES: `{data.smiles}`')

        # Show SMILES visualization
        st.subheader("SMILES Visualization")
        img = Draw.MolToImage(rdkit_mol)
        st.image(img)

        # Load rewiring data with the current parameters
        rewire_data = load_lmdb_data(
            default_db_path,
            data.name,
            epoch_count,
            status,
            ensemble
        )

        if rewire_data:
            # Get the specific key's data
            key = f'{epoch_count}-{status}-{ensemble}-{data.name}'
            if key in rewire_data:
                rewire_info = rewire_data[key]

                # Convert rewiring edges to list of tuples
                rewire_add = rewire_info['rewire_add_edge_index']
                rewire_del = rewire_info['rewire_del_edge_index']
                rewire_add_edges = list(zip(rewire_add[0], rewire_add[1]))
                rewire_del_edges = list(zip(rewire_del[0], rewire_del[1]))

            # Get existing edges
            edge_index = data.edge_index.cpu().numpy()
            existing_edges = list(zip(edge_index[0], edge_index[1]))
            num_nodes = data.x.shape[0]

            # Create atom labels
            atom_labels = {i: f"{i}:{get_atom_symbol(data.x[i])}" for i in range(num_nodes)}

            # Create 2D visualization
            pos_2d = project_3d_to_2d(data.pos)
            pos = {i: (pos_2d[i, 0], pos_2d[i, 1]) for i in range(num_nodes)}

            # Create 2D plot
            st.subheader("2D Structure")
            fig2d = create_2d_plotly(pos, atom_labels, existing_edges,
                                     rewire_add_edges, rewire_del_edges, num_nodes)
            st.plotly_chart(fig2d, use_container_width=True)

            # Create 3D plot
            st.subheader("3D Structure")
            fig3d = create_3d_plotly(data.pos, atom_labels, existing_edges,
                                     rewire_add_edges, rewire_del_edges)
            st.plotly_chart(fig3d, use_container_width=True)

            # Add information about current selection
            st.sidebar.success(f"""
            Current Selection:
            - Epoch: {epoch_count}
            - Status: {status}
            - Ensemble: {ensemble}
            - Molecule: {data.name}
            """)

            # Display edge information
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Atom Mapping:")
                for i in range(num_nodes):
                    st.write(f"Index {i}: `{get_atom_symbol(data.x[i])}`")

            with col2:
                st.subheader("Edge Information:")
                st.write("Original edges:", len(existing_edges))
                st.write("Rewire add edges:", len(rewire_add_edges))
                st.write("Rewire delete edges:", len(rewire_del_edges))
        else:
            st.error("No rewiring data found for this molecule.")
    else:
        st.error('Molecule not found in the QM9 dataset.')
