import streamlit as st
from streamlit.components.v1 import html
import json
import base64
from PIL import Image
from io import BytesIO

def app():
    # Sidebar menu
    tab1, tab2, tab3 = st.tabs([" # Decision Trees", " # Random Forests", "# K-Medoids"])

    with tab1: 
        # Main content area
        st.header("Implémentation de L’algorithme de Decision Trees")

        # Load the notebook
        with open('notebooks/Notebook sur les Arbres de Décision.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                st.markdown(''.join(cell['source']))
            elif cell['cell_type'] == 'code':
                st.code(''.join(cell['source']))

                # Display outputs (if any)
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output['output_type'] == 'stream':
                            # For print outputs
                            st.text(''.join(output['text']))

                        elif output['output_type'] == 'execute_result':
                            if 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))

                        elif output['output_type'] == 'display_data':
                            # Handle images
                            if 'image/png' in output['data']:
                                img_data = output['data']['image/png']
                                img_bytes = base64.b64decode(img_data)
                                image = Image.open(BytesIO(img_bytes))
                                st.image(image)

                            # Handle text/plain if exists
                            elif 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))
        
    with tab2:
        st.header("Implémentation de L’algorithme de Forêts Aléatoires (Random Forests)")
        # Load the notebook
        with open('notebooks/random_forest_fr.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)

            
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                st.markdown(''.join(cell['source']))
            elif cell['cell_type'] == 'code':
                st.code(''.join(cell['source']))

                # Display outputs (if any)
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output['output_type'] == 'stream':
                            # For print outputs
                            st.text(''.join(output['text']))

                        elif output['output_type'] == 'execute_result':
                            if 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))

                        elif output['output_type'] == 'display_data':
                            # Handle images
                            if 'image/png' in output['data']:
                                img_data = output['data']['image/png']
                                img_bytes = base64.b64decode(img_data)
                                image = Image.open(BytesIO(img_bytes))
                                st.image(image)

                            # Handle text/plain if exists
                            elif 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))
        
    with tab3:
        st.header("Implémentation de L’algorithme de Clustering K-Medoids")

        # Load the notebook
        with open('notebooks/KMedoids.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                st.markdown(''.join(cell['source']))
            elif cell['cell_type'] == 'code':
                st.code(''.join(cell['source']))

                # Display outputs (if any)
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output['output_type'] == 'stream':
                            # For print outputs
                            st.text(''.join(output['text']))

                        elif output['output_type'] == 'execute_result':
                            if 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))

                        elif output['output_type'] == 'display_data':
                            # Handle images
                            if 'image/png' in output['data']:
                                img_data = output['data']['image/png']
                                img_bytes = base64.b64decode(img_data)
                                image = Image.open(BytesIO(img_bytes))
                                st.image(image)

                            # Handle text/plain if exists
                            elif 'text/plain' in output['data']:
                                st.text(''.join(output['data']['text/plain']))

