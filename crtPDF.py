import pdfkit
import base64

def create_pdf(content):
    pdf = pdfkit.from_string(content, False)
    b64 = base64.b64encode(pdf).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_kmedoids.pdf">📄 Télécharger en PDF</a>'
    return href

# Add in your app
# st.markdown(create_pdf("<h1>Contenu PDF ici</h1><p>... etc ...</p>"), unsafe_allow_html=True)
