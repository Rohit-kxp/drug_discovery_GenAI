import streamlit as st
import pandas as pd
import io
import time
from utils.molecule_generation import generate_molecules, optimize_molecule
from utils.molecule_properties import predict_properties, predict_admet_properties
from utils.molecule_visualization import display_molecule, display_molecule_grid, display_3d_molecule, display_molecule_comparison
from utils.data_handling import search_pubchem, load_sample_molecules, get_dataset_stats, get_molecules_by_property

st.set_page_config(
    page_title="MoleculeForge AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
 #csss
st.markdown("""
<style>
    /* Main theme colors and styling */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #1abc9c;
        --accent-color: #3498db;
        --text-color: #2c3e50;
        --background-color: #f8f9fa;
        --card-background: white;
    }
    
    /* Header styling */
    .main-header {
        background:  #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-content {
        position: relative;
        z-index: 2;
    }
    
    .animated-bg {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(26, 188, 156, 0.3), rgba(44, 62, 80, 0.2));
        z-index: 1;
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .title-row {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .molecule-icon {
        font-size: 3.5rem;
        margin-right: 1rem;
        animation: molecule-spin 10s linear infinite;
    }
    
    @keyframes molecule-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #f1f1f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Feature list */
    .feature-list {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 1rem;
    }
    
    .feature-item {
        background-color: rgba(255, 255, 255, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        font-weight: 500;
        backdrop-filter: blur(5px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        animation: feature-fade 0.5s ease-in-out;
    }
    
    @keyframes feature-fade {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Cards and UI elements */
    .stButton button {
        background-color: var(--secondary-color) !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #2c3e50, #34495e);
        color: white;
        border-radius: 10px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        z-index: 2;
    }
    
    .footer-animated-bg {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(52, 73, 94, 0.5), rgba(44, 62, 80, 0.5));
        z-index: 1;
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite reverse;
    }
    
    /* Streamlit default element overrides */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    div.stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    div.stTabs [aria-selected="true"] {
        background-color: #1abc9c !important;
        color: white !important;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .subtitle {
            font-size: 1rem;
        }
        .feature-item {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)
#header
st.markdown("""
<div class="main-header">
    <div class="animated-bg"></div>
    <div class="header-content">
        <div class="title-row">
            <div class="molecule-icon">🧪</div>
            <h1 class="main-title">MoleculeForge AI</h1>
        </div>
        <p class="subtitle">Advanced Drug Discovery Platform</p>
        <div class="feature-list">
            <div class="feature-item">Explore</div>
            <div class="feature-item">Generate</div>
            <div class="feature-item">Optimize</div>
            <div class="feature-item">Visualize</div>
            <div class="feature-item">Predict</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Description
st.markdown("""
This platform assists pharmaceutical researchers in drug discovery by leveraging GenAI:
- **Explore** a database of 1.5+ million molecular compounds
- **Generate** novel molecular structures through AI algorithms
- **Optimize** existing molecules for specific pharmaceutical properties
- **Visualize** compounds in interactive 2D and 3D representations
- **Predict** comprehensive molecular properties and ADMET profiles
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Home", "Molecule Generation", "Molecule Optimization", "Property Prediction", "Search Compounds"]
)

# Home page
if app_mode == "Home":
    st.header("Welcome to MoleculeForge")
    with st.spinner("Loading database statistics..."):
        stats = get_dataset_stats()
    
    st.subheader("Database Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Compounds", f"{stats['total_molecules']:,}")
    with col2:
        st.metric("Unique Structures", f"{stats['unique_molecules']:,}")
    with col3:
        if 'qed_mean' in stats:
            st.metric("Average Drug-likeness", f"{stats['qed_mean']:.2f}")
        else:
            st.metric("Database Size", "1.5M+ molecules")
            
    # Property range information
    if 'molecular_weight_min' in stats and 'molecular_weight_max' in stats:
        st.info(f"Molecular Weight Range: {stats['molecular_weight_min']:.1f} - {stats['molecular_weight_max']:.1f} g/mol")
    
    st.write("""
    ### Getting Started
    
    MoleculeForge leverages a massive database of 1.5 million+ chemical structures to provide advanced drug discovery capabilities. Choose a mode from the sidebar to begin:
    
    - **Molecule Generation**: Generate novel drug candidates using AI algorithms
    - **Molecule Optimization**: Fine-tune existing molecules for specific pharmaceutical properties
    - **Property Prediction**: Calculate detailed properties and ADMET profiles
    - **Search Compounds**: Explore our extensive database and find similar structures
    """)
    tab1, tab2 = st.tabs(["Representative Compounds", "Interactive 3D Visualization"])
    
    with tab1:
        st.subheader("High-quality Drug-like Compounds")
        sample_mols = load_sample_molecules(n=8)
        display_molecule_grid(sample_mols, cols=4)
        st.subheader("Property Distribution in Database")
        if 'logp_min' in stats and 'logp_max' in stats:
            st.write(f"**LogP Range:** {stats['logp_min']:.1f} to {stats['logp_max']:.1f}")
        if 'tpsa_min' in stats and 'tpsa_max' in stats:
            st.write(f"**TPSA Range:** {stats['tpsa_min']:.1f} to {stats['tpsa_max']:.1f} Å²")
            
    with tab2:
        st.subheader("Interactive 3D Molecular Visualization")
        st.write("Select a molecule from the sample set to view an interactive 3D model:")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sample_names = [f"Sample {i+1}" for i in range(len(sample_mols))]
            selected_sample = st.selectbox("Choose a molecule", sample_names)
        
            sample_idx = sample_names.index(selected_sample)
            
            viz_style = st.selectbox("Visualization Style", ["stick", "sphere", "line"])
            
            st.info("Rotate, zoom, and explore the molecule in 3D space by interacting with the visualization below.")
        
        with col2:
            display_3d_molecule(sample_mols[sample_idx], style=viz_style)
        
    st.subheader("Technology Behind MolecularForge")
    st.markdown("""
    MoleculeForge AI combines several state-of-the-art technologies:
    
    - **Huge Chemical Database**: Over 1.5 million carefully curated drug-like molecules
    - **Advanced 3D Visualization**: Powered by py3Dmol for interactive exploration
    - **Molecular Property Prediction**: Sophisticated algorithms for ADMET profiling
    - **AI-Driven Optimization**: Genetic algorithms to enhance molecular properties
    
    Our platform helps pharmaceutical researchers accelerate the drug discovery process by providing
    powerful tools to explore chemical space and identify promising candidates efficiently.
    """)

# Molecule Generation
elif app_mode == "Molecule Generation":
    st.header("Generate Novel Molecules")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Generation Parameters")
        num_molecules = st.slider("Number of molecules to generate", 1, 20, 3)
        st.write("Property Constraints (Optional)")
        min_mw = st.number_input("Minimum Molecular Weight", 0.0, 1000.0, 200.0)
        max_mw = st.number_input("Maximum Molecular Weight", 0.0, 1000.0, 500.0)
        
        max_logp = st.slider("Maximum LogP", 0.0, 10.0, 5.0)
        
        include_substructure = st.text_input("Include Substructure (SMILES)", "")
        
        target_property = st.selectbox(
            "Optimize for property",
            ["None", "Solubility", "Bioavailability", "Drug-likeness"]
        )
        
        generate_button = st.button("Generate Molecules")
    
    with col2:
        st.subheader("Generated Molecules")
        
        if generate_button:
            with st.spinner("Generating molecules..."):
                time.sleep(2)
                try:
                    #generate molecules
                    molecules = generate_molecules(
                        num_molecules=num_molecules,
                        min_mw=min_mw,
                        max_mw=max_mw,
                        max_logp=max_logp,
                        include_substructure=include_substructure,
                        target_property=target_property if target_property != "None" else None
                    )
                    
                    if molecules:
                        # Display molecules
                        display_molecule_grid(molecules)
                        
                        # Create a DataFrame for download
                        mol_data = []
                        for i, mol in enumerate(molecules):
                            props = predict_properties(mol)
                            mol_data.append({
                                "ID": f"GEN-{i+1}",
                                "SMILES": mol,
                                **props
                            })
                        
                        df = pd.DataFrame(mol_data)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download Generated Molecules",
                            data=csv,
                            file_name="generated_molecules.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No molecules could be generated with the specified constraints.")
                except Exception as e:
                    st.error(f"An error occurred during molecule generation: {str(e)}")

# Molecule Optimization
elif app_mode == "Molecule Optimization":
    st.header("Optimize Molecules")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Input Molecule")
        
        # Input options
        input_method = st.radio("Input Method", ["SMILES", "Draw"])
        
        if input_method == "SMILES":
            smiles_input = st.text_area("Enter SMILES string", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F")
            if smiles_input:
                st.subheader("Input Molecule Preview")
                display_molecule(smiles_input)
        else:
            st.info("Drawing functionality coming soon. Please use SMILES input for now.")
            smiles_input = ""
        
        st.subheader("Optimization Parameters")
        
        target_property = st.selectbox(
            "Optimize for property",
            ["Solubility", "Bioavailability", "Drug-likeness", "Blood-brain barrier penetration"]
        )
        
        optimization_strength = st.slider("Optimization Strength", 0.1, 1.0, 0.5)
        
        maintain_similarity = st.slider("Maintain Similarity to Original", 0.1, 1.0, 0.7)
        
        # Optimize button
        optimize_button = st.button("Optimize Molecule")
    
    with col2:
        if smiles_input and optimize_button:
            st.subheader("Optimization Results")
            
            with st.spinner("Optimizing molecule..."):
                # Add slight delay to simulate computation
                time.sleep(2)
                
                try:
                    # Optimize molecule
                    optimized_molecules = optimize_molecule(
                        smiles_input,
                        target_property=target_property,
                        optimization_strength=optimization_strength,
                        similarity_threshold=maintain_similarity
                    )
                    
                    if optimized_molecules:
                        # Display original vs optimized
                        all_mols = [smiles_input] + optimized_molecules
                        labels = ["Original"] + [f"Optimized {i+1}" for i in range(len(optimized_molecules))]
                        
                        # Get properties for each molecule
                        all_props = [predict_properties(mol) for mol in all_mols]
                        
                        # Create comparison table
                        comparison_data = []
                        for i, (mol, props, label) in enumerate(zip(all_mols, all_props, labels)):
                            comparison_data.append({
                                "Molecule": label,
                                "SMILES": mol,
                                **props
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.write(comparison_df)
                        
                        # Display molecules
                        st.subheader("Molecule Visualization")
                        display_molecule_grid(all_mols, labels=labels)
                        
                        # Add 3D visualization and comparison
                        with st.expander(f"Show 3D view for Molecule {i+1}"):
                                    # viz_style = st.radio(f"Visualization Style for Molecule {i+1}", 
                                    #                        ["stick", "sphere", "line"]
                                    #             )
                            # viz_style = "stick"
                                    # viz_style = change_style(viz_style)
                            # display_3d_molecule(smiles, style=viz_style)
                        # show_3d = st.checkbox("Show 3D Visualization and Comparison")
                        
                            # Use the molecule comparison function with 3D support
                            st.subheader("Advanced Molecular Visualization")
                            
                            # Select molecules to compare
                            col3d1, col3d2 = st.columns(2)
                            with col3d1:
                                orig_idx = 0  # Original molecule
                                st.write(f"Original: {labels[orig_idx]}")
                            with col3d2:
                                opt_idx = st.selectbox("Select optimized molecule for comparison:", 
                                                      labels[1:], 
                                                      index=0) 
                                # Convert back to list index
                                opt_idx = labels.index(opt_idx)
                            
                            # Create property changes dictionary for comparison
                            property_changes = {}
                            for prop in all_props[0].keys():
                                if prop in all_props[opt_idx]:
                                    property_changes[prop] = {
                                        "original": all_props[orig_idx][prop],
                                        "new": all_props[opt_idx][prop]
                                    }
                            
                            # Display the advanced comparison with 3D
                            display_molecule_comparison(
                                all_mols[orig_idx], 
                                all_mols[opt_idx],
                                property_changes
                            )
                        
                        # Download button
                        csv = comparison_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            data=csv,
                            file_name="optimization_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Optimization failed. Please try different parameters.")
                except Exception as e:
                    st.error(f"An error occurred during optimization: {str(e)}")

# Property Prediction
elif app_mode == "Property Prediction":
    st.header("Predict Molecular Properties")
    
    # Input options
    input_method = st.radio("Input Method", ["SMILES", "File Upload"])
    
    if input_method == "SMILES":
        smiles_input = st.text_area("Enter SMILES string (one per line for multiple molecules)")
        
        if smiles_input:
            molecules = [smiles.strip() for smiles in smiles_input.split("\n") if smiles.strip()]
            
            if st.button("Predict Properties"):
                if molecules:
                    results = []
                    
                    with st.spinner("Predicting properties..."):
                        for i, smiles in enumerate(molecules):
                            try:
                                # Predict properties
                                props = predict_properties(smiles)
                                
                                # Add to results
                                results.append({
                                    "Molecule ID": i+1,
                                    "SMILES": smiles,
                                    **props
                                })
                                
                                # Display molecule
                                st.subheader(f"Molecule {i+1}")
                                display_molecule(smiles)
                                
                                # Add 3D view option
                                # show_3d = st.checkbox(f"Show 3D view for Molecule {i+1}")
                                # def change_style(viz_style):
                                #     viz_style = st.radio("Visualization Style for Molecule",["stick","sphere","line"])
                                #     return viz_style
                                
                                with st.expander(f"Show 3D view for Molecule {i+1}"):
                                    # viz_style = st.radio(f"Visualization Style for Molecule {i+1}", 
                                    #                        ["stick", "sphere", "line"]
                                    #             )
                                    viz_style = "stick"
                                    # viz_style = change_style(viz_style)
                                    display_3d_molecule(smiles, style=viz_style)
                                
                                # Display properties
                                for prop, value in props.items():
                                    st.write(f"**{prop}:** {value}")
                                # st.write(smiles)
                                # st.write(predict_admet_properties(smiles))
                                # Display ADMET properties
                                # show_admet = st.checkbox(f"Show detailed ADMET predictions for Molecule {i+1}")
                                # if show_admet:
                                with st.expander("ADMET Predictions"):
                                    admet_props = predict_admet_properties(smiles)
                                    for category, values in admet_props.items():
                                        st.markdown(f"**{category}**")
                                        for prop, value in values.items():
                                            st.write(f"• {prop}: {value}")

       
    

                                # if show_admet:
                                #     st.subheader("ADMET Predictions")
                                #     admet_props = predict_admet_properties(smiles)
                                #     for category, values in admet_props.items():
                                #         st.write(f"**{category}**")
                                #         for prop, value in values.items():
                                #             st.write(f"  • {prop}: {value}")
                                
                                # st.divider()
                            except Exception as e:
                                st.error(f"Error processing molecule {i+1}: {str(e)}")
                    
                    # Create DataFrame and enable download
                    if results:
                        results_df = pd.DataFrame(results)
                        st.subheader("Summary Table")
                        st.write(results_df)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            data=csv,
                            file_name="property_predictions.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No valid SMILES strings provided.")
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload CSV or Excel file with SMILES column", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                # Determine file type and read
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Check for SMILES column
                st.write("Preview of uploaded data:")
                st.write(df.head())
                
                # Select SMILES column
                smiles_col = st.selectbox("Select SMILES column", df.columns)
                
                if st.button("Predict Properties"):
                    if smiles_col in df.columns:
                        molecules = df[smiles_col].tolist()
                        results = []
                        
                        with st.spinner("Predicting properties..."):
                            for i, smiles in enumerate(molecules):
                                try:
                                    # Predict properties
                                    props = predict_properties(smiles)
                                    
                                    # Add row data and properties
                                    result_row = {col: df.iloc[i][col] for col in df.columns}
                                    result_row.update(props)
                                    results.append(result_row)
                                except Exception as e:
                                    st.error(f"Error processing row {i+1}: {str(e)}")
                        
                        # Create DataFrame and enable download
                        if results:
                            results_df = pd.DataFrame(results)
                            st.subheader("Results")
                            st.write(results_df)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Results",
                                data=csv,
                                file_name="property_predictions.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("SMILES column not found in the file.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Search Compounds
elif app_mode == "Search Compounds":
    st.header("Search for Compounds")
    
    search_options = st.radio("Search by", ["Name", "SMILES", "Substructure"])
    
    query = st.text_input("Enter search query")
    
    if query and st.button("Search"):
        with st.spinner("Searching compounds..."):
            try:
                # Search compounds
                search_results = search_pubchem(query, search_type=search_options.lower())
                
                if search_results:
                    st.success(f"Found {len(search_results)} compounds")
                    
                    for i, result in enumerate(search_results):
                        st.subheader(f"Compound {i+1}: {result.get('name', 'Unknown')}")
                        
                        
                        
                        
                            # Display molecule
                        display_molecule(result['smiles'])
                            
                            # Add 3D view option for search results
                        with st.expander(f"Show 3D view for Compound {i+1}"):
    
                            # viz_style = st.selectbox(f"Visualization Style for Compound {i+1}", 
                            #                        ["stick", "sphere", "line", "cartoon"])
                            display_3d_molecule(result['smiles'], style="stick")
                        
                        
                            # Display properties
                        st.write(f"**SMILES:** {result['smiles']}")
                        for prop, value in result.items():
                            if prop not in ['name', 'smiles']:
                                st.write(f"**{prop}:** {value}")
                            
                            # Add advanced property predictions
                        with st.expander(f"Show advanced property predictions for Compound {i+1}"):
                            
                            with st.spinner("Calculating detailed properties..."):
                                    # Basic properties
                                props = predict_properties(result['smiles'])
                                st.subheader("Calculated Properties")
                                for prop, value in props.items():
                                    st.write(f"**{prop}:** {value}")
                                        
                                    # ADMET properties
                                admet = predict_admet_properties(result['smiles'])
                                st.subheader("ADMET Predictions")
                                for category, values in admet.items():
                                    st.write(f"**{category}**")
                                    for prop, value in values.items():
                                        st.write(f"  • {prop}: {value}")
                    
                    # Create DataFrame and enable download
                    results_df = pd.DataFrame(search_results)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Search Results",
                        data=csv,
                        file_name="search_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No compounds found matching your query.")
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

#footer
st.markdown("""
<div style="background-color: #2c3e50; padding: 20px; color: white; border-radius: 10px; text-align: center;">
    <div>
        <strong style="font-size: 1.2em;">MoleculeForge AI</strong><br>
        <span>Advanced Drug Discovery Platform</span>
    </div>
</div>
""", unsafe_allow_html=True)

