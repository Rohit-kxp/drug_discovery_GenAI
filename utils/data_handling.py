import pandas as pd
import requests
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
import streamlit as st
import random
import os
from rdkit.Chem import QED
from . import dataset_handler

# Global variables for dataset management
_GLOBAL_DATASET = None
_DATASET_SIZE = 1500000  # 15 lakh molecules

@st.cache_resource
def get_global_dataset():
    
    global _GLOBAL_DATASET
    if _GLOBAL_DATASET is None:
        try:
            # First, add a loading animation
            with st.spinner("Loading molecular database..."):
                # Check for custom dataset first
                custom_dataset = "../attached_assets/drug_discovery_100k.csv"
                if os.path.exists(custom_dataset):
                    try:
                        st.info("Loading custom drug discovery dataset...")
                        _GLOBAL_DATASET = dataset_handler.load_csv_dataset(custom_dataset)
                        st.success(f"Database ready with {len(_GLOBAL_DATASET):,} molecules")
                    except Exception as e:
                        st.error(f"Error loading custom dataset: {str(e)}")
                        st.info("Falling back to default database...")
                
                # If custom dataset failed or wasn't found, try ChEMBL
                if _GLOBAL_DATASET is None or len(_GLOBAL_DATASET) == 0:
                    # Check if we have available storage for large database
                    storage_available = os.path.exists('./data/cache') and os.access('./data/cache', os.W_OK)
                    
                    if storage_available:
                        # st.info("Preparing molecular database...")
                        _GLOBAL_DATASET = dataset_handler.load_chembl_dataset(limit=100000)
                        
                        if len(_GLOBAL_DATASET) > 0:
                            st.success(f"Database loaded with {len(_GLOBAL_DATASET):,} molecules")
                        else:
                            # If ChEMBL fails, load a synthetic dataset
                            # st.info("Initializing molecular database...")
                            _GLOBAL_DATASET = dataset_handler.create_synthetic_dataset(size=100000)
                            # st.success(f"Database ready with {len(_GLOBAL_DATASET):,} molecules (representing a 1.5M compound library)")
                    else:
                        # If storage is limited, create a smaller synthetic dataset
                        # st.info("Initializing molecular database...")
                        _GLOBAL_DATASET = dataset_handler.create_synthetic_dataset(size=50000)
                        # st.success(f"Database ready with {len(_GLOBAL_DATASET):,} molecules")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            # Create a minimal dataset if all else fails
            fallback_smiles = load_fallback_molecules()
            _GLOBAL_DATASET = pd.DataFrame({
                "molecule_id": [f"MOL_{i+1}" for i in range(len(fallback_smiles))],
                "smiles": fallback_smiles,
                "molecular_weight": [round(Descriptors.MolWt(Chem.MolFromSmiles(s)), 2) for s in fallback_smiles], # type: ignore
                "logp": [round(Descriptors.MolLogP(Chem.MolFromSmiles(s)), 2) for s in fallback_smiles], # type: ignore
                "tpsa": [round(Descriptors.TPSA(Chem.MolFromSmiles(s)), 2) for s in fallback_smiles], # type: ignore
                "hbd": [Descriptors.NumHDonors(Chem.MolFromSmiles(s)) for s in fallback_smiles], # type: ignore
                "hba": [Descriptors.NumHAcceptors(Chem.MolFromSmiles(s)) for s in fallback_smiles], # type: ignore
                "qed": [round(QED.qed(Chem.MolFromSmiles(s)), 3) for s in fallback_smiles], 
            })
            st.warning("Using a limited dataset of drug molecules. For full functionality, please restart the app.")
    
    return _GLOBAL_DATASET

def load_fallback_molecules():
    """Load a set of reliable sample drug molecules as fallback"""
    sample_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",  # Amoxicillin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
        "CN1C(=O)CN=C(C2=C1C=CC=C2)C3=CC=CC=C3Cl",  # Diazepam
        "CN(C)C(=N)NC(=N)N",  # Metformin
        "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",  # Fluoxetine
        "C1=CC=C2C(=C1)C=CC=C2C3=CC=CC=C3",  # Biphenyl
        "CCC1(CC)C(=O)NC(=O)NC1=O"  # Phenobarbital
    ]
    return sample_smiles

@st.cache_data(ttl=600)  # Cache for 10 minutes
def search_pubchem(query, search_type="name", max_results=10):
    """
    Search for compounds in the dataset or from PubChem
    
    Parameters:
    - query: The search query
    - search_type: One of 'name', 'smiles', or 'substructure'
    - max_results: Maximum number of results to return
    
    Returns:
    - List of dictionaries with compound data
    """
    df = get_global_dataset()
    results = []
    
    try:
        # Search in our dataset first
        if search_type == "name":
            # Search by molecule ID or any text fields
            search_fields = ["molecule_id"]
            if "name" in df.columns:
                search_fields.append("name")
                
            matches_df = dataset_handler.search_molecules(df, query, search_in=search_fields, limit=max_results)
            
        elif search_type == "smiles":
            # Exact SMILES match
            matches_df = df[df["smiles"] == query].head(max_results)
            
            # If no exact match, try substructure search
            if len(matches_df) == 0 and Chem.MolFromSmiles(query):
                # This is simplified - in a real app, we'd use a proper substructure search
                # For demonstration, we'll just do a text search in SMILES
                matches_df = dataset_handler.search_molecules(df, query, search_in=["smiles"], limit=max_results)
                
        elif search_type == "substructure":
            # Simplified substructure search using text matching
            # In a real application, this would use RDKit's substructure search
            matches_df = dataset_handler.search_molecules(df, query, search_in=["smiles"], limit=max_results)
            
        else:
            matches_df = pd.DataFrame()
            
        # Convert search results to list of dictionaries
        if len(matches_df) > 0:
            for _, row in matches_df.iterrows():
                result = {
                    "molecule_id": row["molecule_id"],
                    "smiles": row["smiles"]
                }
                
                # Add any additional columns that exist
                for col in row.index:
                    if col not in ["molecule_id", "smiles"]:
                        result[col] = row[col]
                        
                # Make sure we have a name field
                if "name" not in result:
                    result["name"] = f"Compound {result['molecule_id']}"
                    
                results.append(result)
                
        # If we didn't find enough results, try fetching from PubChem
        if len(results) < max_results and search_type == "name":
            # Use PubChem REST API
            try:
                pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
                response = requests.get(pubchem_url)
                
                if response.status_code == 200:
                    data = response.json()
                    compounds = data["PropertyTable"]["Properties"]
                    
                    for compound in compounds:
                        # Check if we've reached the limit
                        if len(results) >= max_results:
                            break
                            
                        # Create result dictionary
                        result = {
                            "name": query,
                            "molecule_id": f"PUBCHEM_{compound.get('CID', 'unknown')}",
                            "smiles": compound.get("CanonicalSMILES", ""),
                            "molecular_formula": compound.get("MolecularFormula", ""),
                            "molecular_weight": compound.get("MolecularWeight", 0)
                        }
                        
                        # Calculate additional properties if possible
                        try:
                            mol = Chem.MolFromSmiles(result["smiles"])
                            if mol:
                                result["logp"] = round(Descriptors.MolLogP(mol), 2) # type: ignore
                                result["tpsa"] = round(Descriptors.TPSA(mol), 2) # type: ignore
                                result["hbd"] = Descriptors.NumHDonors(mol) # type: ignore
                                result["hba"] = Descriptors.NumHAcceptors(mol) # type: ignore
                        except:
                            pass
                            
                        results.append(result)
            except:
                # If PubChem API fails, we'll just use what we have from our dataset
                pass
    except Exception as e:
        st.error(f"Search error: {str(e)}")
    
    return results

@st.cache_data(ttl=3600)
def load_sample_molecules(n=8):
    """
    Load a set of sample drug molecules for display
    
    Parameters:
    - n: Number of molecules to return
    
    Returns:
    - List of SMILES strings
    """
    df = get_global_dataset()
    
    # Get high-quality drug-like molecules
    if 'qed' in df.columns:
        # Sort by drug-likeness (QED) if available
        sample_df = df.sort_values(by='qed', ascending=False).head(n)
    else:
        # Otherwise just take the first n rows
        sample_df = df.head(n)
    
    # Extract SMILES strings
    sample_smiles = sample_df['smiles'].tolist()
    
    # If we have fewer molecules than requested, fill with fallback molecules
    if len(sample_smiles) < n:
        sample_smiles.extend(load_fallback_molecules()[:n-len(sample_smiles)])
    
    return sample_smiles

def export_results_to_csv(data, filename="results.csv"):
    """
    Convert data to CSV format
    
    Parameters:
    - data: List of dictionaries or DataFrame
    - filename: Name of the output file
    
    Returns:
    - CSV string
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    csv = df.to_csv(index=False)
    return csv

def get_molecules_by_property(property_name, min_val=None, max_val=None, n=1000):
    """
    Get molecules filtered by a specific property
    
    Parameters:
    - property_name: Name of the property to filter by
    - min_val: Minimum value (inclusive)
    - max_val: Maximum value (inclusive)
    - n: Maximum number of molecules to return
    
    Returns:
    - DataFrame with filtered molecules
    """
    df = get_global_dataset()
    
    return dataset_handler.generate_subset_by_property(df, property_name, min_val, max_val, n)

def get_dataset_stats():
    """
    Get statistics about the dataset
    
    Returns:
    - Dictionary with dataset statistics
    """
    df = get_global_dataset()
    
    stats = {
        "total_molecules": len(df),
        "unique_molecules": df["smiles"].nunique()
    }
    
    # Add property ranges if available
    for prop in ["molecular_weight", "logp", "tpsa", "qed"]:
        if prop in df.columns:
            stats[f"{prop}_min"] = round(df[prop].min(), 2)
            stats[f"{prop}_max"] = round(df[prop].max(), 2)
            stats[f"{prop}_mean"] = round(df[prop].mean(), 2)
    
    return stats
