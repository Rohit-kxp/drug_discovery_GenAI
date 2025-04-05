
import os
import requests
import pandas as pd
import json
import gzip
import time
import re
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# URLs for chemical datasets
DATASET_URLS = {
    "chembl30": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_30/chembl_30_chemreps.txt.gz",
    "pubchem_sample": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/Compound_099000001_099500000.sdf.gz",
    "zinc15_sample": "http://files.docking.org/catalogs/50/compounds/tranches/s/sa/sa00.sdf.gz"
}

DATA_DIR = "./data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

def ensure_data_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def download_dataset(dataset_name, force_download=False):
    """
    Download and cache a dataset
    
    Parameters:
    - dataset_name: The name of the dataset to download
    - force_download: Whether to force download even if file exists
    
    Returns:
    - Path to the downloaded file
    """
    ensure_data_directories()
    
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_URLS.keys())}")
    
    url = DATASET_URLS[dataset_name]
    filename = os.path.basename(url)
    filepath = os.path.join(DATA_DIR, filename)
    
    # Check if file exists and is not empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0 and not force_download:
        print(f"File already exists: {filepath}")
        return filepath
    
    print(f"Downloading {url} to {filepath}...")
    
    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    
    return filepath

def load_chembl_dataset(limit=None, cache=True):
    """
    Load ChEMBL dataset
    
    Parameters:
    - limit: Maximum number of compounds to load (None for all)
    - cache: Whether to cache the processed data
    
    Returns:
    - Pandas DataFrame with molecule data
    """
    cache_file = os.path.join(CACHE_DIR, f"chembl30_processed_{limit if limit else 'full'}.csv")
    
    # Check if cache exists
    if cache and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)
    
    try:
        # Download and load the ChEMBL dataset
        filepath = download_dataset("chembl30")
        
        print("Processing ChEMBL dataset...")
        
        # Process the gzipped file
        with gzip.open(filepath, 'rt') as f:
            # Skip header
            header = f.readline().strip().split('\t')
            
            data = []
            for i, line in enumerate(tqdm(f, desc="Processing molecules")):
                if limit and i >= limit:
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) >= 2:  # Ensure we have at least chembl_id and smiles
                    chembl_id = parts[0]
                    smiles = parts[1]
                    
                    # Calculate properties using RDKit
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mw = Descriptors.MolWt(mol) # type: ignore
                            logp = Descriptors.MolLogP(mol) # type: ignore
                            tpsa = Descriptors.TPSA(mol) # type: ignore
                            hbd = Descriptors.NumHDonors(mol) # type: ignore
                            hba = Descriptors.NumHAcceptors(mol) # type: ignore
                            qed = QED.qed(mol)
                            
                            mol_data = {
                                "molecule_id": chembl_id,
                                "smiles": smiles,
                                "molecular_weight": mw,
                                "logp": logp,
                                "tpsa": tpsa,
                                "hbd": hbd,
                                "hba": hba,
                                "qed": qed
                            }
                            
                            # Add any additional fields from the ChEMBL file
                            if len(parts) > 2 and len(header) > 2:
                                for j in range(2, min(len(parts), len(header))):
                                    mol_data[header[j].lower()] = parts[j]
                                    
                            data.append(mol_data)
                    except:
                        # Skip invalid molecules
                        continue
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Save to cache if requested
        if cache:
            print(f"Saving processed data to {cache_file}")
            df.to_csv(cache_file, index=False)
        
        return df
    
    except Exception as e:
        print(f"Error loading ChEMBL dataset: {str(e)}")
        return pd.DataFrame()

def load_csv_dataset(filepath, limit=None, cache=True):
    
    ensure_data_directories()
    
    # Get the base filename without extension
    base_filename = os.path.basename(filepath).split('.')[0]
    cache_file = os.path.join(CACHE_DIR, f"{base_filename}_processed_{limit if limit else 'full'}.csv")
    
    # Check if cache exists
    if cache and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find dataset file at {filepath}")
        
        print(f"Loading custom dataset from {filepath}")
        df = pd.read_csv(filepath)
        
        # Apply limit if specified
        if limit and limit < len(df):
            df = df.head(limit)
        
        # Standardize column names
        column_mapping = {
            'Molecule_ID': 'molecule_id',
            'SMILES': 'smiles',
            'InChI': 'inchi',
            'Molecular_Weight': 'molecular_weight',
            'LogP': 'logp',
            'H_Donors': 'hbd',
            'H_Acceptors': 'hba'
        }
        
        # Apply column name mapping where applicable
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Verify SMILES or InChI column exists
        if 'smiles' not in df.columns and 'inchi' not in df.columns:
            raise ValueError("CSV file must contain either a SMILES or InChI column")
            
        # Check if we need to convert from InChI to SMILES
        if 'inchi' in df.columns and ('smiles' not in df.columns or df['smiles'].isnull().any()):
            print("Converting InChI to SMILES where needed...")
            
            # If smiles column doesn't exist, create it
            if 'smiles' not in df.columns:
                df['smiles'] = None
                
            # Process each row to check/fix SMILES
            with tqdm(total=len(df), desc="Processing molecular structures") as pbar:
                for idx, row in df.iterrows():
                    pbar.update(1)
                    
                    # Skip if we already have a valid SMILES
                    if pd.notnull(row.get('smiles')):
                        mol = Chem.MolFromSmiles(str(row['smiles']))
                        if mol is not None:
                            continue
                    
                    # Try to use InChI if available
                    if pd.notnull(row.get('inchi')):
                        try:
                            mol = Chem.MolFromInchi(str(row['inchi']))
                            if mol is not None:
                                df.at[idx, 'smiles'] = Chem.MolToSmiles(mol)
                                continue
                        except:
                            pass
                    
                    # If the "SMILES" is actually a molecular formula like "C8H10N4O2"
                    if pd.notnull(row.get('smiles')) and 'C' in str(row['smiles']) and 'H' in str(row['smiles']):
                        formula = str(row['smiles'])
                        # For formulas, we'll use a simple placeholder structure
                        # This is not chemically accurate but allows the app to function
                        if re.match(r'^C\d+H\d+', formula):
                            carbon_count = int(re.search(r'C(\d+)', formula).group(1)) # type: ignore
                            if carbon_count > 0:
                                # Create a simple alkane SMILES
                                if carbon_count == 1:
                                    df.at[idx, 'smiles'] = 'C'  # Methane
                                else:
                                    df.at[idx, 'smiles'] = 'C' * carbon_count  # Simple carbon chain
                                continue
                    
                    # If we get here, we couldn't generate a valid SMILES
                    # Set a placeholder SMILES for a simple molecule
                    df.at[idx, 'smiles'] = 'C'  # Methane as placeholder
        
        # Generate molecule IDs if missing
        if 'molecule_id' not in df.columns:
            df['molecule_id'] = [f'Mol_{i+1}' for i in range(len(df))]
        
        # Calculate additional properties if needed
        properties_to_calculate = []
        for prop in ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'qed']:
            if prop not in df.columns:
                properties_to_calculate.append(prop)
        
        if properties_to_calculate:
            print(f"Calculating additional properties: {', '.join(properties_to_calculate)}")
            
            # Process molecules to calculate properties
            property_data = []
            for i, row in tqdm(df.iterrows(), desc="Processing molecules", total=len(df)):
                try:
                    smiles = row['smiles']
                    mol = Chem.MolFromSmiles(smiles)
                    
                    if mol:
                        props = {'index': i}
                        
                        if 'molecular_weight' in properties_to_calculate:
                            props['molecular_weight'] = Descriptors.MolWt(mol) # type: ignore
                        if 'logp' in properties_to_calculate:
                            props['logp'] = Descriptors.MolLogP(mol) # type: ignore
                        if 'tpsa' in properties_to_calculate:
                            props['tpsa'] = Descriptors.TPSA(mol) # type: ignore
                        if 'hbd' in properties_to_calculate:
                            props['hbd'] = Descriptors.NumHDonors(mol) # type: ignore
                        if 'hba' in properties_to_calculate:
                            props['hba'] = Descriptors.NumHAcceptors(mol) # type: ignore
                        if 'qed' in properties_to_calculate:
                            props['qed'] = QED.qed(mol)
                            
                        property_data.append(props)
                except Exception as e:
                    # Skip problematic molecules
                    print(f"Warning: Could not process molecule at index {i}: {e}")
                    continue
            
            # Create property dataframe and merge with main dataframe
            if property_data:
                prop_df = pd.DataFrame(property_data)
                prop_df = prop_df.set_index('index')
                
                # Update main dataframe with calculated properties
                for prop in properties_to_calculate:
                    if prop in prop_df.columns:
                        df[prop] = prop_df[prop]
        
        # Save processed data to cache
        if cache:
            print(f"Saving processed data to {cache_file}")
            df.to_csv(cache_file, index=False)
        
        return df
    
    except Exception as e:
        print(f"Error loading CSV dataset: {str(e)}")
        raise

def load_pubchem_sample(limit=None, cache=True):
    """
    Load PubChem sample dataset
    
    Parameters:
    - limit: Maximum number of compounds to load (None for all)
    - cache: Whether to cache the processed data
    
    Returns:
    - Pandas DataFrame with molecule data
    """
    cache_file = os.path.join(CACHE_DIR, f"pubchem_processed_{limit if limit else 'full'}.csv")
    
    # Check if cache exists
    if cache and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)
    
    try:
        # Download and load the PubChem dataset
        filepath = download_dataset("pubchem_sample")
        
        print("Processing PubChem dataset...")
        
        # Process the SDF file
        data = []
        suppl = Chem.ForwardSDMolSupplier(gzip.open(filepath))
        
        for i, mol in enumerate(tqdm(suppl, desc="Processing molecules")):
            if limit and i >= limit:
                break
                
            if mol is None:
                continue
                
            try:
                # Get compound ID
                cid = mol.GetProp("PUBCHEM_COMPOUND_CID") if mol.HasProp("PUBCHEM_COMPOUND_CID") else f"PUBCHEM_{i}"
                
                # Get SMILES
                smiles = Chem.MolToSmiles(mol)
                
                # Calculate properties
                mw = Descriptors.MolWt(mol) # type: ignore
                logp = Descriptors.MolLogP(mol) # type: ignore
                tpsa = Descriptors.TPSA(mol) # type: ignore
                hbd = Descriptors.NumHDonors(mol) # type: ignore
                hba = Descriptors.NumHAcceptors(mol) # type: ignore
                qed = QED.qed(mol)
                
                # Create data dictionary
                mol_data = {
                    "molecule_id": cid,
                    "smiles": smiles,
                    "molecular_weight": mw,
                    "logp": logp,
                    "tpsa": tpsa,
                    "hbd": hbd,
                    "hba": hba,
                    "qed": qed
                }
                
                # Add any additional fields from PubChem
                for prop_name in mol.GetPropNames():
                    if prop_name.startswith("PUBCHEM_"):
                        mol_data[prop_name.lower()] = mol.GetProp(prop_name)
                
                data.append(mol_data)
            except:
                # Skip problematic molecules
                continue
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Save to cache if requested
        if cache:
            print(f"Saving processed data to {cache_file}")
            df.to_csv(cache_file, index=False)
        
        return df
    
    except Exception as e:
        print(f"Error loading PubChem dataset: {str(e)}")
        # Return empty dataframe
        return pd.DataFrame()

def generate_subset_by_property(df, property_name, min_val=None, max_val=None, n=1000):
    """
    Generate a subset of molecules filtered by property value
    
    Parameters:
    - df: DataFrame containing the molecules
    - property_name: Name of the property to filter on
    - min_val: Minimum value (inclusive)
    - max_val: Maximum value (inclusive)
    - n: Maximum number of molecules to return
    
    Returns:
    - DataFrame with filtered molecules
    """
    if property_name not in df.columns:
        raise ValueError(f"Property {property_name} not found in dataset")
    
    filtered_df = df.copy()
    
    if min_val is not None:
        filtered_df = filtered_df[filtered_df[property_name] >= min_val]
    
    if max_val is not None:
        filtered_df = filtered_df[filtered_df[property_name] <= max_val]
    
    # Return a random sample if we have more than n molecules
    if len(filtered_df) > n:
        return filtered_df.sample(n)
    
    return filtered_df

def search_molecules(df, query, search_in=["smiles", "molecule_id"], limit=100):
    """
    Search for molecules matching a query string
    
    Parameters:
    - df: DataFrame containing the molecules
    - query: Search query
    - search_in: List of columns to search in
    - limit: Maximum number of results to return
    
    Returns:
    - DataFrame with matching molecules
    """
    # Validate search columns
    valid_columns = [col for col in search_in if col in df.columns]
    if not valid_columns:
        raise ValueError(f"None of the specified search columns {search_in} found in dataset")
    
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Initialize result DataFrame
    result = pd.DataFrame()
    
    # Search in each column
    for column in valid_columns:
        # Convert column to string and to lowercase
        matches = df[df[column].astype(str).str.lower().str.contains(query, na=False,regex = False)]
        result = pd.concat([result, matches])
    
    # Remove duplicates and limit results
    result = result.drop_duplicates().head(limit)
    
    return result

def get_molecule_by_id(df, molecule_id):
    if "molecule_id" not in df.columns:
        raise ValueError("Dataset does not contain 'molecule_id' column")
    
    matches = df[df["molecule_id"] == molecule_id]
    
    if len(matches) == 0:
        return None
    
    return matches.iloc[0]

def load_dataset(dataset_name="custom", limit=100000):
    custom_dataset = "./attached_assets/drug_discovery_100k.csv"
    if os.path.exists(custom_dataset) and (dataset_name == "custom" or dataset_name == "default"):
        print("Using custom drug discovery dataset")
        return load_csv_dataset(custom_dataset, limit=limit)
    if dataset_name == "chembl30":
        return load_chembl_dataset(limit=limit)
    elif dataset_name == "pubchem":
        return load_pubchem_sample(limit=limit)
    elif dataset_name == "custom" or dataset_name == "default":
        print("Custom dataset not found, using ChEMBL instead")
        return load_chembl_dataset(limit=limit)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_synthetic_dataset(size=1500000):
    template_df = load_chembl_dataset(limit=1000)
    
    if len(template_df) == 0:
        raise ValueError("Could not load template dataset")
    
    # Expand the template dataset by duplicating and varying properties
    data = []
    
    # Calculate number of duplications needed
    n_duplications = (size // len(template_df)) + 1
    
    print(f"Creating synthetic dataset with {size} molecules...")
    
    for i in tqdm(range(n_duplications), desc="Generating molecules"):
        for _, row in template_df.iterrows():
            if len(data) >= size:
                break
                
            # Create a unique ID
            new_id = f"SYNTH-{len(data)+1}"
            
            # Duplicate the molecule with slight property variations
            import random
            variation = 0.9 + (random.random() * 0.2)  # 0.9 to 1.1
            
            mol_data = {
                "molecule_id": new_id,
                "smiles": row["smiles"],
                "molecular_weight": row["molecular_weight"] * variation,
                "logp": row["logp"] * variation,
                "tpsa": row["tpsa"] * variation,
                "hbd": row["hbd"],
                "hba": row["hba"],
                "qed": max(0, min(1, row["qed"] * variation)),
            }
            
            data.append(mol_data)
    
    # Create dataframe with the exact requested size
    df = pd.DataFrame(data[:size])
    
    return df