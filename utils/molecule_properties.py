from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
import numpy as np

def predict_properties(smiles):
    """
    Predict molecular properties for the given SMILES string
    
    Parameters:
    - smiles: SMILES representation of the molecule
    
    Returns:
    - Dictionary of predicted properties
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Calculate basic descriptors
    mw = Descriptors.MolWt(mol) # type: ignore
    logp = Descriptors.MolLogP(mol) # type: ignore
    tpsa = Descriptors.TPSA(mol) # type: ignore
    hba = Descriptors.NumHAcceptors(mol) # type: ignore
    hbd = Descriptors.NumHDonors(mol) # type: ignore
    rotatable_bonds = Descriptors.NumRotatableBonds(mol) # type: ignore
    aromatic_rings = Chem.Lipinski.NumAromaticRings(mol) # type: ignore
    heavy_atoms = mol.GetNumHeavyAtoms()
    
    # Lipinski's Rule of Five
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5 : lipinski_violations += 1
    if hba > 10: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    
    # Drug-likeness
    qed = QED.qed(mol)
    
    # Synthetic accessibility (simplified approximation)
    # In a real application, more sophisticated models would be used
    synth_accessibility = min(10, max(1, 10 - 0.1 * mol.GetNumAtoms()))
    
    # Solubility estimation (simplified)
    # LogS = 0.5 - logP * 0.01 - MW * 0.0025 (simplified approximation)
    logS = 0.5 - logp * 0.01 - mw * 0.0025
    solubility_class = "Highly soluble" if logS > 0 else "Moderately soluble" if logS > -2 else "Poorly soluble"
    
    # Blood-brain barrier penetration (simplified rule-based)
    # Rule: logP < 3 and MW < 400 and TPSA < 90 and HBD < 3
    bbb_penetration = (logp < 3 and mw < 400 and tpsa < 90 and hbd < 3)
    bbb_class = "High" if bbb_penetration else "Low"
    
    # Bioavailability score (simplified)
    # Based on Lipinski violations
    if lipinski_violations == 0:
        bioavailability = "High"
    elif lipinski_violations == 1:
        bioavailability = "Medium"
    else:
        bioavailability = "Low"
    
    return {
        "Molecular Weight": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "H-Bond Acceptors": hba,
        "H-Bond Donors": hbd,
        "Rotatable Bonds": rotatable_bonds,
        "Aromatic Rings": aromatic_rings,
        "Heavy Atoms": heavy_atoms,
        "Lipinski Violations": lipinski_violations,
        "Drug-likeness (QED)": round(qed, 3),
        "Synthetic Accessibility": round(synth_accessibility, 1),
        "Solubility Class": solubility_class,
        "BBB Penetration": bbb_class,
        "Bioavailability": bioavailability
    }

def predict_admet_properties(smiles):
    """
    Predict ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity)
    
    In a real application, this would use specialized machine learning models.
    This implementation uses rule-based predictions based on physiochemical properties.
    
    Parameters:
    - smiles: SMILES representation of the molecule
    
    Returns:
    - Dictionary of predicted ADMET properties by category
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Get basic properties for rules
    mw = Descriptors.MolWt(mol) # type: ignore
    logp = Descriptors.MolLogP(mol) # type: ignore
    tpsa = Descriptors.TPSA(mol) # type: ignore
    hba = Descriptors.NumHAcceptors(mol) # type: ignore
    hbd = Descriptors.NumHDonors(mol) # type: ignore
    rotatable_bonds = Descriptors.NumRotatableBonds(mol) # type: ignore
    aromatic_rings = Chem.Lipinski.NumAromaticRings(mol) # type: ignore
    qed = QED.qed(mol)
    
    # Absorption properties
    absorption = {}
    
    # Caco-2 permeability prediction
    # Rule: Good permeability if TPSA < 90 and HBD < 3
    if tpsa < 90 and hbd < 3:
        absorption["Caco-2 Permeability"] = "High"
    elif tpsa < 120 and hbd < 5:
        absorption["Caco-2 Permeability"] = "Medium"
    else:
        absorption["Caco-2 Permeability"] = "Low"
    
    # Human Intestinal Absorption (HIA)
    # Rule: Good absorption if MW < 500, TPSA < 140, rotatable bonds < 10
    if mw < 500 and tpsa < 140 and rotatable_bonds < 10:
        absorption["Human Intestinal Absorption"] = "High (>80%)"
    elif mw < 600 and tpsa < 180:
        absorption["Human Intestinal Absorption"] = "Medium (50-80%)"
    else:
        absorption["Human Intestinal Absorption"] = "Low (<50%)"
    
    # P-glycoprotein substrate
    # Rule: Likely substrate if MW > 400 and TPSA > 90
    if mw > 400 and tpsa > 90:
        absorption["P-glycoprotein Substrate"] = "Likely"
    else:
        absorption["P-glycoprotein Substrate"] = "Unlikely"
    
    # Distribution properties
    distribution = {}
    
    # Blood-Brain Barrier (BBB) penetration
    if logp < 3 and mw < 400 and tpsa < 70 and hbd < 3:
        distribution["BBB Penetration"] = "High"
    elif logp < 4 and mw < 450 and tpsa < 90:
        distribution["BBB Penetration"] = "Medium"
    else:
        distribution["BBB Penetration"] = "Low"
    
    # Volume of distribution
    # Rule: High if logP > 3, Low if logP < 0
    if logp > 3:
        distribution["Volume of Distribution"] = "High (>3 L/kg)"
    elif logp > 0:
        distribution["Volume of Distribution"] = "Medium (1-3 L/kg)"
    else:
        distribution["Volume of Distribution"] = "Low (<1 L/kg)"
    
    # Plasma protein binding
    # Rule: High binding if logP > 3 or rings > 3
    if logp > 3 or aromatic_rings > 3:
        distribution["Plasma Protein Binding"] = "High (>90%)"
    elif logp > 1 or aromatic_rings > 1:
        distribution["Plasma Protein Binding"] = "Medium (70-90%)"
    else:
        distribution["Plasma Protein Binding"] = "Low (<70%)"
    
    # Metabolism properties
    metabolism = {}
    
    # CYP450 substrate likelihood
    # Rules based on structural features
    if logp > 3 and aromatic_rings >= 2:
        metabolism["CYP450 Substrate"] = "Likely"
    else:
        metabolism["CYP450 Substrate"] = "Less likely"
    
    # Metabolic stability
    # Rule: Stability decreases with increasing logP and rotatable bonds
    stability_score = 5 - min(5, max(0, (logp-2)/1.5 + rotatable_bonds/5))
    
    if stability_score > 3.5:
        metabolism["Metabolic Stability"] = "High"
    elif stability_score > 2:
        metabolism["Metabolic Stability"] = "Medium"
    else:
        metabolism["Metabolic Stability"] = "Low"
    
    # CYP450 inhibition (simplified)
    if logp > 4 and mw > 400:
        metabolism["CYP450 Inhibition"] = "Likely"
    else:
        metabolism["CYP450 Inhibition"] = "Less likely"
    
    # Excretion properties
    excretion = {}
    
    # Total clearance prediction
    # Rule: Higher clearance with lower MW and logP
    if mw < 350 and logp < 3:
        excretion["Total Clearance"] = "High"
    elif mw < 500 and logp < 5:
        excretion["Total Clearance"] = "Medium"
    else:
        excretion["Total Clearance"] = "Low"
    
    # Renal clearance
    # Rule: Higher renal clearance with lower logP (more hydrophilic)
    if logp < 1:
        excretion["Renal Clearance"] = "High"
    elif logp < 3:
        excretion["Renal Clearance"] = "Medium"
    else:
        excretion["Renal Clearance"] = "Low"
    
    # Half-life estimation
    # Rule: Longer half-life with higher MW, logP, and protein binding
    half_life_score = min(5, (mw/100) * 0.3 + logp * 0.4 + (1 if logp > 3 else 0) * 0.3)
    
    if half_life_score > 3.5:
        excretion["Half-life"] = "Long (>24h)"
    elif half_life_score > 2:
        excretion["Half-life"] = "Medium (8-24h)"
    else:
        excretion["Half-life"] = "Short (<8h)"
    
    # Toxicity properties
    toxicity = {}
    
    # Rule-based toxicity predictions
    # hERG inhibition (cardiac toxicity)
    if (logp > 3.7 and mw > 250) or (logp > 4 and hba > 5):
        toxicity["hERG Inhibition Risk"] = "High"
    elif logp > 3 or mw > 400:
        toxicity["hERG Inhibition Risk"] = "Medium"
    else:
        toxicity["hERG Inhibition Risk"] = "Low"
    
    # Hepatotoxicity
    if logp > 3 and mw > 300:
        toxicity["Hepatotoxicity Risk"] = "Medium"
    elif logp > 5 or mw > 500:
        toxicity["Hepatotoxicity Risk"] = "High"
    else:
        toxicity["Hepatotoxicity Risk"] = "Low"
    
    # Mutagenicity (simplified rule - real prediction requires structural alerts)
    # This is a very simplified approximation
    toxicity["Mutagenicity Risk"] = "Requires structural analysis" 
    
    # Drug-likeness based toxicity interpretation
    if qed > 0.7:
        toxicity["Overall Toxicity Risk"] = "Low"
    elif qed > 0.5:
        toxicity["Overall Toxicity Risk"] = "Medium"
    else:
        toxicity["Overall Toxicity Risk"] = "High"
    
    return {
        "Absorption": absorption,
        "Distribution": distribution,
        "Metabolism": metabolism,
        "Excretion": excretion,
        "Toxicity": toxicity
    }
