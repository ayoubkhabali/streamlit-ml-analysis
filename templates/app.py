import streamlit as st
from streamlit.components.v1 import html
import json
import base64
from PIL import Image
from io import BytesIO

def app():
    # Sidebar menu
    # st.sidebar.title('Example Algorithm Selection')
    algorithm = st.sidebar.radio(
        'Choose an algorithm:',
        ['Decision Trees', 'Random Forests', 'K-Medoids'], 
    )

    # Main content area
    if algorithm == 'Decision Trees':
        st.header('Decision Trees')

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
        
    elif algorithm == 'Random Forests':
        st.header('Random Forests')
        
    else:  # K-Medoids
        st.header('K-Medoids')

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
