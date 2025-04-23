import streamlit as st

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
        
    elif algorithm == 'Random Forests':
        st.header('Random Forests')
        
    else:  # K-Medoids
        st.header('K-Medoids')