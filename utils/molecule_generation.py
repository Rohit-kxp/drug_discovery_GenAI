import random
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen, Lipinski, QED
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs
import os
import numpy as np

# List of common molecular fragments
FRAGMENTS = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1cccnc1",  # pyridine (another form)
    "c1ccco1",   # furan
    "c1cccs1",   # thiophene
    "C1CCNCC1",  # piperidine
    "C1CCOCC1",  # tetrahydropyran
    "C1CCOC1",   # tetrahydrofuran
    "CC(=O)O",   # acetic acid
    "CCO",       # ethanol
    "CCN",       # ethylamine
    "CC=O",      # acetaldehyde
    "CNC=O",     # N-methylformamide
    "CN(C)C=O",  # dimethylformamide
    "CC#N",      # acetonitrile
    "CS(=O)(=O)O", # methanesulfonic acid
    "CF",        # fluoromethane
    "CCl",       # chloromethane
    "CBr",       # bromomethane
    "CI",        # iodomethane
    "CC(F)(F)F", # trifluoromethane
]

# Linkers to connect fragments
LINKERS = [
    "-",         # single bond
    "=",         # double bond
    "#",         # triple bond
    "-C-",       # methylene bridge
    "-N-",       # amine bridge
    "-O-",       # ether bridge
    "-S-",       # thioether bridge
    "-C(=O)-",   # ketone bridge
    "-C(=O)O-",  # ester bridge
    "-C(=O)N-",  # amide bridge
    "-N=N-",     # azo bridge
    "-CH=CH-",   # alkene bridge
]

# Functional groups to add
FUNCTIONAL_GROUPS = [
    "F",         # fluoro
    "Cl",        # chloro
    "Br",        # bromo
    "I",         # iodo
    "O",         # hydroxy (after attaching)
    "N",         # amino (after attaching)
    "C(=O)O",    # carboxylic acid
    "C(=O)N",    # amide
    "S(=O)(=O)O", # sulfonic acid
    "C#N",       # nitrile
    "N=O",       # nitroso
    "N(=O)=O",   # nitro
]

def is_valid_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_properties(smiles):
    ##Calculate basic molecular properties
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        "mw": Descriptors.MolWt(mol), # type: ignore
        "logp": Descriptors.MolLogP(mol), # type: ignore
        "hba": Descriptors.NumHAcceptors(mol), # type: ignore
        "hbd": Descriptors.NumHDonors(mol), # type: ignore
        "tpsa": Descriptors.TPSA(mol), # type: ignore
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol) # type: ignore
    }
    
    return properties

def meets_constraints(smiles, min_mw=0.0, max_mw=float('inf'), max_logp=None, include_substructure=None):
    """Check if molecule meets the specified constraints"""
    props = calculate_properties(smiles)
    if props is None:
        return False
    
    # Check molecular weight
    if props["mw"] < min_mw or props["mw"] > max_mw:
        return False
    
    # Check LogP
    if max_logp is not None and props["logp"] > max_logp:
        return False
    
    # Check substructure
    if include_substructure:
        mol = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmiles(include_substructure)
        if pattern and mol:
            if not mol.HasSubstructMatch(pattern):
                return False
    
    return True

def generate_random_molecule():
    ##Generate random molecule
    # Randomly select number of fragments
    num_fragments = random.randint(1, 3)
    
    # Select random fragments
    selected_fragments = random.sample(FRAGMENTS, num_fragments)
    
    # If only one fragment, return it possibly with functional group
    if num_fragments == 1:
        smiles = selected_fragments[0]
        if random.random() > 0.5:
            func_group = random.choice(FUNCTIONAL_GROUPS)
            smiles = f"{smiles}{func_group}"
        return smiles
    
    # Otherwise connect fragments with linkers
    molecule = selected_fragments[0]
    for i in range(1, num_fragments):
        linker = random.choice(LINKERS)
        molecule += linker + selected_fragments[i]
    
    # Maybe add a functional group
    if random.random() > 0.7:
        func_group = random.choice(FUNCTIONAL_GROUPS)
        molecule += func_group
    
    return molecule

def generate_molecules(num_molecules=5, min_mw=0.0, max_mw=float('inf'), 
                      max_logp=None, include_substructure=None, target_property=None):
   
    valid_molecules = []
    attempts = 0
    max_attempts = num_molecules * 50  # Limit the number of attempts
    
    while len(valid_molecules) < num_molecules and attempts < max_attempts:
        attempts += 1
        
        # Generate a new molecule
        smiles = generate_random_molecule()
        
        # Check if it's valid and meets constraints
        if is_valid_molecule(smiles) and meets_constraints(
            smiles, min_mw, max_mw, max_logp, include_substructure
        ):
            # Add only if not already in the list
            if smiles not in valid_molecules:
                valid_molecules.append(smiles)
    
    return valid_molecules

def calculate_molecular_similarity(smiles1, smiles2):
    
    #Calculate the Tanimoto similarity between two molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    # Generate Morgan fingerprints for the molecules
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1 = fp_gen.GetFingerprint(mol1)
    fp2 = fp_gen.GetFingerprint(mol2)
    
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return similarity

def calculate_property_score(mol, target_property):
    """
    Calculate a score for how well a molecule matches the target property
    
    Parameters:
    - mol: RDKit molecule object
    - target_property: The property to optimize for (e.g., 'Solubility', 'Drug-likeness')
    
    Returns:
    - Score between 0 and 1
    """
    if target_property == "Solubility":
        # For solubility, use a combination of LogP and TPSA
        logp = Crippen.MolLogP(mol) # type: ignore
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        
        # Lower LogP and higher TPSA generally indicate better solubility
        # Convert to a 0-1 score where 1 is more soluble
        logp_score = max(0, min(1, (5 - logp) / 5))  # LogP < 0 is good for solubility
        tpsa_score = min(1, tpsa / 150)  # Higher TPSA is good for solubility
        
        return (logp_score + tpsa_score) / 2
    
    elif target_property == "Bioavailability":
        # For bioavailability, use QED and Lipinski's Rule of Five violations
        qed_score = QED.qed(mol)
        lipinski_violations = 0
        
        mw = Descriptors.MolWt(mol) # type: ignore
        logp = Descriptors.MolLogP(mol) # type: ignore
        h_donors = Descriptors.NumHDonors(mol) # type: ignore
        h_acceptors = Descriptors.NumHAcceptors(mol) # type: ignore
        
        if mw > 500: lipinski_violations += 1
        if logp > 5: lipinski_violations += 1
        if h_donors > 5: lipinski_violations += 1
        if h_acceptors > 10: lipinski_violations += 1
        
        lipinski_score = (4 - lipinski_violations) / 4
        
        return (qed_score + lipinski_score) / 2
    
    elif target_property == "Drug-likeness":
        # For drug-likeness, use QED (Quantitative Estimate of Drug-likeness)
        return QED.qed(mol)
    
    elif target_property == "Blood-brain barrier penetration":
        # For BBB penetration, use a combination of molecular weight, LogP, and TPSA
        mw = Descriptors.MolWt(mol) # type: ignore
        logp = Descriptors.MolLogP(mol) # type: ignore
        tpsa = rdMolDescriptors.CalcTPSA(mol) # type: ignore
        h_donors = Descriptors.NumHDonors(mol) # type: ignore
        
        # Rule: logP < 3 and MW < 400 and TPSA < 90 and HBD < 3
        mw_score = 1 if mw < 400 else (450 - mw) / 50 if mw < 450 else 0
        logp_score = 1 if logp < 3 else (4 - logp) if logp < 4 else 0
        tpsa_score = 1 if tpsa < 90 else (120 - tpsa) / 30 if tpsa < 120 else 0
        hbd_score = 1 if h_donors < 3 else (4 - h_donors) if h_donors < 4 else 0
        
        return (mw_score + logp_score + tpsa_score + hbd_score) / 4
    
    # Default if no specific property is targeted
    return QED.qed(mol)  # Default to drug-likeness

def mutate_molecule(mol):
    """
    Apply a random mutation to a molecule
    
    Parameters:
    - mol: RDKit molecule object
    
    Returns:
    - SMILES string of the mutated molecule, or None if mutation failed
    """
    # Create a copy of the molecule to modify
    mol = Chem.Mol(mol)
    
    # Choose a mutation type
    mutation_type = random.choice([
        "add_functional_group",
        "replace_atom",
        "add_ring",
        "remove_group"
    ])
    
    try:
        if mutation_type == "add_functional_group":
            # Add a functional group to a random atom
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return None
                
            atom_idx = random.randint(0, num_atoms - 1)
            func_group = random.choice(FUNCTIONAL_GROUPS)
            
            # For simplicity, we'll convert to SMILES and back as a crude mutation
            smiles = Chem.MolToSmiles(mol)
            new_smiles = smiles + func_group
            new_mol = Chem.MolFromSmiles(new_smiles)
            
            if new_mol:
                return Chem.MolToSmiles(new_mol)
            
        elif mutation_type == "replace_atom":
            # Replace a random atom with a different element
            # This is a simplified example - in reality, this would need much more sophisticated handling
            smiles = Chem.MolToSmiles(mol)
            
            # Randomly modify a carbon to nitrogen or oxygen
            if 'C' in smiles:
                replacement = random.choice(['N', 'O'])
                pos = random.choice([i for i, c in enumerate(smiles) if c == 'C'])
                new_smiles = smiles[:pos] + replacement + smiles[pos+1:]
                new_mol = Chem.MolFromSmiles(new_smiles)
                
                if new_mol:
                    return Chem.MolToSmiles(new_mol)
            
        elif mutation_type == "add_ring":
            # Add a ring structure
            fragment = random.choice([f for f in FRAGMENTS if 'c1' in f or 'C1' in f])
            linker = random.choice(LINKERS)
            
            smiles = Chem.MolToSmiles(mol)
            new_smiles = smiles + linker + fragment
            new_mol = Chem.MolFromSmiles(new_smiles)
            
            if new_mol:
                return Chem.MolToSmiles(new_mol)
                
        elif mutation_type == "remove_group":
            # This is a complex operation that would require substructure matching and removal
            # For simplicity, we'll just convert to SMILES and try a cruder approach
            smiles = Chem.MolToSmiles(mol)
            
            # Try to identify and remove a simple functional group like F, Cl, Br, I
            for fg in ['F', 'Cl', 'Br', 'I', 'OH', 'NH2']:
                if fg in smiles:
                    pos = smiles.find(fg)
                    if pos > 0:  # Make sure we're not at the beginning
                        new_smiles = smiles[:pos] + smiles[pos+len(fg):]
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        if new_mol:
                            return Chem.MolToSmiles(new_mol)
    
    except Exception as e:
        # If any error occurs during mutation, return None
        pass
        
    return None

def optimize_molecule(smiles, target_property=None, optimization_strength=0.5, similarity_threshold=0.7):
    """
    Optimize a molecule for a target property while maintaining similarity
    
    Parameters:
    - smiles: The SMILES string of the molecule to optimize
    - target_property: The property to optimize for
    - optimization_strength: How much to prioritize optimization over similarity (0-1)
    - similarity_threshold: Minimum similarity to maintain with the original molecule
    
    Returns:
    - List of optimized molecule SMILES strings
    """
    if not is_valid_molecule(smiles):
        return []
    
    # Create the original molecule
    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return []
    
    # Calculate original property score
    orig_score = calculate_property_score(orig_mol, target_property) if target_property else 0.5
    
    # Number of variants to generate and number of iterations
    num_variants = 5
    num_iterations = 50
    population_size = 20
    
    # Initialize population with the original molecule and some random variants
    population = [smiles]
    
    # Generate an initial diverse population
    for _ in range(population_size - 1):
        new_smiles = generate_random_molecule()
        if is_valid_molecule(new_smiles):
            population.append(new_smiles)
    
    # Evolutionary optimization
    for iteration in range(num_iterations):
        # Calculate fitness for each molecule in the population
        fitness_scores = []
        
        for mol_smiles in population:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                fitness_scores.append(0)
                continue
                
            # Calculate property score
            prop_score = calculate_property_score(mol, target_property) if target_property else 0.5
            
            # Calculate similarity to original
            sim_score = calculate_molecular_similarity(smiles, mol_smiles)
            
            # Combined fitness score - weighted average of property improvement and similarity
            property_weight = optimization_strength
            similarity_weight = 1 - optimization_strength
            
            fitness = property_weight * prop_score + similarity_weight * sim_score
            
            # Penalize if similarity is below threshold
            if sim_score < similarity_threshold:
                fitness *= (sim_score / similarity_threshold)
                
            fitness_scores.append(fitness)
        
        # Create new generation
        new_population = []
        
        # Elitism - keep the best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:3]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Generate new individuals through mutation
        while len(new_population) < population_size:
            # Select a parent based on fitness (tournament selection)
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            parent_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            parent_smiles = population[parent_idx]
            
            # Mutate the parent
            parent_mol = Chem.MolFromSmiles(parent_smiles)
            if parent_mol:
                child_smiles = mutate_molecule(parent_mol)
                if child_smiles and is_valid_molecule(child_smiles) and child_smiles not in new_population:
                    new_population.append(child_smiles)
        
        # Update population
        population = new_population
    
    # Select the best variants
    population_with_scores = []
    for mol_smiles in population:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            continue
            
        # Calculate property score
        prop_score = calculate_property_score(mol, target_property) if target_property else 0.5
        
        # Calculate similarity to original
        sim_score = calculate_molecular_similarity(smiles, mol_smiles)
        
        # Only include if similarity is above threshold
        if sim_score >= similarity_threshold:
            # Combined score for ranking
            combined_score = optimization_strength * prop_score + (1 - optimization_strength) * sim_score
            population_with_scores.append((mol_smiles, combined_score, prop_score, sim_score))
    
    # Sort by combined score
    population_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top unique variants
    result = []
    seen = set([smiles])  # Include original to avoid duplicates
    
    for mol_smiles, _, _, _ in population_with_scores:
        if mol_smiles not in seen and len(result) < num_variants:
            result.append(mol_smiles)
            seen.add(mol_smiles)
    
    return result
