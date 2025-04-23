import streamlit as st
from streamlit_option_menu import option_menu
import templates.app, templates.DescriptionAlgorithm

st.title('Machine Learning Algorithms Analysis: ')

# Main application class
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            # st.image("assets/image.png", use_container_width=True)
            # Using Markdown for styling
            app = option_menu(
                menu_title='Menu',
                options=["Description Algorithm", 'Algorithm' ],
                icons=['house-fill', "cloud-arrow-up", "gear", "building-fill-gear", "card-text"],
                menu_icon='list',
                default_index=0,
            )

        # Route to the selected page
        if app == "Algorithm":
            templates.app.app()
        elif app == "Description Algorithm":
            templates.DescriptionAlgorithm.app()


# Display main application
st.markdown(
    """
    <style> 
    .st-emotion-cache-yw8pof.ekr3hml4 {
        max-width: none;
    }
    </style>
    """, unsafe_allow_html=True
)
app = MultiApp()
app.run()
