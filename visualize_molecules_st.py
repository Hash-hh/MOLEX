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

# Set page config to narrow layout
st.set_page_config(layout="centered", page_title="Molecule Visualizer")
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


def create_3d_plotly(pos_3d, atom_labels, edges, custom_edges):
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
        marker=dict(size=10,
                    color=[atom_colors.get(atom_labels[i].split(':')[1], 'lightblue') for i in range(len(atom_labels))],
                    line_width=1,
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

    # Add custom edges
    for edge in custom_edges:
        fig.add_trace(go.Scatter3d(
            x=[pos_3d_np[edge[0], 0], pos_3d_np[edge[1], 0]],
            y=[pos_3d_np[edge[0], 1], pos_3d_np[edge[1], 1]],
            z=[pos_3d_np[edge[0], 2], pos_3d_np[edge[1], 2]],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
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


def create_2d_plotly(pos, atom_labels, existing_edges, custom_edges, original_node_count):
    """Create an interactive 2D plot using Plotly"""
    # Create edges traces
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

    # Add custom edges
    for edge in custom_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color='green', dash='dash'),
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


def get_atom_symbol(features):
    """Convert QM9 one-hot encoded features to atom symbol"""
    atom_types = ['H', 'C', 'N', 'O', 'F']
    idx = features[:5].argmax().item()
    return atom_types[idx]


dataset = load_dataset()

st.title('MOLEX - Molecule Explorer')

name_or_smiles = st.text_input('Enter molecule name or SMILES', 'gdb_26883')

st.write("Enter edges in COO format (two rows of comma-separated indices)")
row1 = st.text_input('Source nodes:', '0, 0, 5, 5, 5')
row2 = st.text_input('Target nodes:', '8, 9, 6, 7, 8')

if name_or_smiles:
    with st.spinner('Searching for molecule...'):
        data, rdkit_mol = find_molecule(name_or_smiles, dataset)

    if data is not None and rdkit_mol is not None:
        st.write(f'SMILES: `{data.smiles}`')

        # Show SMILES visualization
        st.subheader("SMILES Visualization")
        img = Draw.MolToImage(rdkit_mol)
        st.image(img)

        edge_index = data.edge_index.cpu().numpy()
        existing_edges = list(zip(edge_index[0], edge_index[1]))

        # Get number of nodes from the data
        num_nodes = data.x.shape[0]

        # Create atom labels
        atom_labels = {i: f"{i}:{get_atom_symbol(data.x[i])}" for i in range(num_nodes)}

        try:
            source_nodes = [int(x.strip()) for x in row1.split(',') if x.strip()]
            target_nodes = [int(x.strip()) for x in row2.split(',') if x.strip()]

            if len(source_nodes) != len(target_nodes):
                st.error("Source and target node lists must have the same length!")
            else:
                # Validate node indices
                invalid_nodes = [n for n in source_nodes + target_nodes if n >= num_nodes]  #  checking if all the nodes are within the range of the number of nodes (+ here is concatenation)
                if invalid_nodes:
                    st.error(f"Invalid node indices: {invalid_nodes}. Available nodes are 0-{num_nodes - 1}")
                else:
                    custom_edges = list(zip(source_nodes, target_nodes))

                    # Project QM9 3D coordinates to 2D
                    pos_2d = project_3d_to_2d(data.pos)
                    # pos_2d = pos_2d * 10  # Scale up for better visualization (use if getting an img)
                    pos = {i: (pos_2d[i, 0], pos_2d[i, 1]) for i in range(num_nodes)}

                    # Create 2D interactive plot
                    st.subheader("2D Structure")
                    fig2d = create_2d_plotly(pos, atom_labels, existing_edges, custom_edges, num_nodes)
                    st.plotly_chart(fig2d, use_container_width=True)

                    # Create 3D interactive plot
                    st.subheader("3D Structure")
                    fig3d = create_3d_plotly(data.pos, atom_labels, existing_edges, custom_edges)
                    st.plotly_chart(fig3d, use_container_width=True)

                    # Display atom mapping and edge information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Atom Mapping:")
                        for i in range(num_nodes):
                            st.write(f"Index {i}: {get_atom_symbol(data.x[i])}")

                    with col2:
                        st.write("Edge Information:")
                        st.write("Number of original edges:", len(existing_edges))
                        st.write("Number of custom edges:", len(custom_edges))

        except ValueError as e:
            st.error("Invalid input format. Please enter comma-separated integers.")
    else:
        st.error('Molecule not found in the QM9 dataset.')
