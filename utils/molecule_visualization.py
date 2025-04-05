import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem # type: ignore
from rdkit.Chem.Draw import rdMolDraw2D # type: ignore
from io import BytesIO
import base64
import math
import re
import py3Dmol
import json

def mol_to_svg(mol, molSize=(300, 300), kekulize=True, drawer=None, **kwargs):
    if mol is None:
        return None
    
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
    
    if kekulize:
        try:
            Chem.Kekulize(mol)
        except:
            mol = Chem.Mol(mol)
    
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    
    drawer.DrawMolecule(mol, **kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg

def display_molecule(smiles, size=(300, 300), legend=""):
    if not smiles:
        st.warning("No SMILES string provided.")
        return
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error(f"Invalid SMILES string: {smiles}")
            return
        
        svg = mol_to_svg(mol, molSize=size)
        st.image(svg) # type: ignore
        
        if legend:
            st.caption(legend)
    except Exception as e:
        st.error(f"Error displaying molecule: {str(e)}")

def display_molecule_grid(smiles_list, cols=3, size=(200, 200), labels=None):
    if not smiles_list:
        st.warning("No molecules to display.")
        return
    
    # Calculate number of rows based on number of molecules and columns
    n_mols = len(smiles_list)
    n_rows = math.ceil(n_mols / cols)
    
    # Create a list of valid molecules
    mols = []
    legends = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                if labels is not None and i < len(labels):
                    legends.append(labels[i])
                else:
                    legends.append(f"Molecule {i+1}")
        except:
            st.warning(f"Could not process molecule: {smiles}")
    
    if not mols:
        st.error("No valid molecules to display.")
        return
    
    # Create grid
    for row in range(n_rows):
        cols_for_row = st.columns(cols)
        
        for col in range(cols):
            idx = row * cols + col
            
            if idx < len(mols):
                with cols_for_row[col]:
                    svg = mol_to_svg(mols[idx], molSize=size)
                    st.image(svg) # type: ignore
                    st.caption(f"{legends[idx]}\nSMILES: {smiles_list[idx]}")


def display_3d_molecule(smiles, width=500, height=400, style="stick"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG()) #type: ignore
    AllChem.UFFOptimizeMolecule(mol) #type:ignore

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=400, height=300)
    viewer.addModel(mol_block, 'mol')
    viewer.setStyle({style: {}})
    viewer.zoomTo()

    html = viewer._make_html()
    st.components.v1.html(html, height=300) # type: ignore
    
def display_molecule_comparison(original_smiles, modified_smiles, property_changes=None):
    if not original_smiles or not modified_smiles:
        st.warning("Both original and modified SMILES strings are required.")
        return
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Molecule")
        display_molecule(original_smiles)
    
    with col2:
        st.subheader("Modified Molecule")
        display_molecule(modified_smiles)
    
    # Display 3D option
    
    st.subheader("3D Visualization")
    view_type = "Both"
    style = "stick"
        
    if view_type == "Original" or view_type == "Both":
        st.subheader("Original Molecule (3D)")
        display_3d_molecule(original_smiles, style=style)
        
    if view_type == "Modified" or view_type == "Both":
        st.subheader("Modified Molecule (3D)")
        display_3d_molecule(modified_smiles, style=style)
    if property_changes:
        st.subheader("Property Changes")
        changes_cols = st.columns(len(property_changes))
        for i, (prop, values) in enumerate(property_changes.items()):
            with changes_cols[i]:
                st.metric(
                    
                    label=prop,
                    value=values["new"],
                    delta=values["new"] - values["original"]
                )
