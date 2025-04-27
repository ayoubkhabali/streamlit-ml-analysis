import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import os

def show_rf_notebook():
    """
    Affiche un notebook Jupyter pour la démonstration de l'algorithme Forêt Aléatoire
    """
    st.header("Notebook Interactif sur les Forêts Aléatoires")
    
    # Chemin vers votre notebook
    notebook_path = "templates/random_forest/notebook/random_forest_fr.ipynb"
    
    st.write("""
    Ce notebook interactif démontre l'algorithme des Forêts Aléatoires avec des visualisations et des exemples de code.
    Vous pouvez voir les détails d'implémentation et expérimenter avec différents paramètres.
    """)
    
    # Convertir le notebook en HTML
    if os.path.exists(notebook_path):
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = nbformat.read(f, as_version=4)
        
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, resources) = html_exporter.from_notebook_node(notebook_content)
        
        # Afficher le HTML
        st.components.v1.html(body, width=None, height=800, scrolling=True)
    else:
        st.error(f"Le fichier notebook n'a pas été trouvé: {notebook_path}")

if __name__ == "__main__":
    show_rf_notebook()