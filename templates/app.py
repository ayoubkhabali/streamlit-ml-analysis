import streamlit as st
import templates.random_forest.app as random_forest_app
from templates.random_forest.src.rf_notebook import show_rf_notebook


def app(show_notebook=False):
    # Sidebar menu
    # st.sidebar.title('Example Algorithm Selection')
    algorithm = st.sidebar.radio(
        'Choose an algorithm:',
        ['Decision Trees', 'Random Forests', 'K-Medoids'], 
    )

    # If notebook is selected and algorithm is Random Forests
    if show_notebook:
        if algorithm == 'Random Forests':
            show_rf_notebook()
        else:
            st.warning("Notebook view is only available for Random Forests algorithm.")
            st.info("Please select Random Forests algorithm to view the notebook.")
        return

    # Main content area
    if algorithm == 'Decision Trees':
        st.header('Decision Trees')
        
    elif algorithm == 'Random Forests':
        random_forest_app.main()
        
    else:  # K-Medoids
        st.header('K-Medoids')