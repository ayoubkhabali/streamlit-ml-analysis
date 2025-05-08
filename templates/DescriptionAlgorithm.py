import streamlit as st
from PIL import Image

def app():
    # Sidebar menu
    tab1, tab2, tab3 = st.tabs([" # Decision Trees", " # Random Forests", "# K-Medoids"])

    with tab1: 
        # Main content area
        st.header("L’algorithme de Decision Trees")
        
        with st.expander('Description', True):
            st.markdown("## 1. Description de l’algorithme Decision Trees")

            st.markdown("""
            Les arbres de décision sont des modèles d'apprentissage automatique supervisé qui permettent de prendre des décisions en suivant une série de règles simples organisées de manière hiérarchique. Ils sont particulièrement appréciés pour leur interprétabilité et leur capacité à traiter aussi bien des problèmes de classification que de régression.
            
            Un arbre de décision se présente sous forme d'un graphe arborescent où:
            - Chaque nœud interne représente un test sur un attribut
            - Chaque branche représente un résultat du test
            - Chaque feuille représente une décision finale (classe ou valeur)
            """)
        
        with st.expander('Caractéristiques', True):
            st.markdown("## 2. Caractéristiques de Decision Trees")

            st.markdown("""
                - Tree-like structure with nodes and branches  
                - Root node, internal nodes, and leaf nodes  
                - Binary or multi-way splits  
                - Hierarchical decision making process  
                - Handles both numerical and categorical data  
                - No need for feature scaling or normalization  
                - Easy to interpret and visualize  
                - Prone to overfitting if not pruned  
                - Uses impurity measures (e.g., Gini, entropy) for splits  
                - Capable of modeling non-linear relationships  
                - Fast training on small to medium datasets  
                - Sensitive to small changes in data (unstable)
            """)

        with st.expander('Mode de fonctionnement', True):
            st.markdown("## 3. Mode de fonctionnement de Decision Trees")

            st.markdown("### Critères de division pour arbres de décision")

            st.markdown("- **Indice de Gini**: Mesure l'impureté d'un nœud")
            st.latex(r"Gini(t) = 1 - \sum_{i=1}^{c} p(i|t)^2")
            st.markdown("où $p(i|t)$ est la proportion d'observations de classe $i$ dans le nœud $t$.")

            st.markdown("- **Entropie et Gain d'information**: Mesure de la réduction de l'incertitude")
            st.latex(r"Entropie(t) = -\sum_{i=1}^{c} p(i|t) \log_2 p(i|t)")
            st.latex(r"GainInfo(t, a) = Entropie(t) - \sum_{v \in \text{Valeurs}(a)} \frac{|t_v|}{|t|} Entropie(t_v)")

            st.markdown("### Pour les problèmes de régression:")

            st.markdown("- **Réduction de la variance**:")
            st.latex(r"Variance(t) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2")
            st.latex(r"Reduction(t, a) = Variance(t) - \sum_{v \in \text{Valeurs}(a)} \frac{|t_v|}{|t|} Variance(t_v)")

            st.markdown("### Élagage (pruning)")
            st.markdown("L'élagage est une technique pour réduire la complexité et éviter le surapprentissage :")
            st.markdown("- **Pré-élagage**: Arrêter la croissance de l'arbre avant qu'il ne devienne trop spécifique")
            st.markdown("- **Post-élagage**: Construire l'arbre complet, puis éliminer les branches peu significatives")

        with st.expander('Avantages et Inconvénients', True):   
            st.markdown("## 4. Avantages et Inconvénients")

            st.markdown("### Avantages")
            st.markdown("""
                - Facilement interprétables (modèle "boîte blanche")
                - Peu de préparation des données requise
                - Capable de traiter des données numériques et catégorielles
                - Gestion naturelle des valeurs manquantes
                - Performance calculatoire efficace
            """)

            st.markdown("### Inconvénients")
            st.markdown("""
                - Tendance au surapprentissage (overfitting)
                - Instabilité (sensibilité aux petites variations dans les données)
                - Biais vers les attributs avec plus de niveaux
            """)
        
        with st.expander('Domaines et champs d\'application', True):
            st.markdown("## 5. Domaines et champs d'application")

            st.markdown("""
                - **Medical diagnosis**
                - **Customer churn prediction**
                - **Credit risk assessment**
                - **Plant or animal species classification**
            """)

    with tab2:
        st.header("L’algorithme de Forêts Aléatoires (Random Forests)")
        
        with st.expander('Description', True):
            st.markdown("## 1. Description de l’algorithme Random Forests")

            st.markdown("""
            Les forêts aléatoires représentent une avancée majeure dans le domaine de l'apprentissage automatique.
            Elles reposent sur l'idée de combiner plusieurs arbres de décision pour améliorer la robustesse, 
            la précision et la capacité de généralisation des modèles prédictifs. Grâce à leur principe de vote majoritaire 
            ou de moyenne des prédictions, elles corrigent efficacement les faiblesses inhérentes aux arbres individuels.
            """)
            st.markdown("""
            Une forêt aléatoire est un ensemble d'arbres de décision, chacun construit à partir d'un échantillon 
            aléatoire des données et d'un sous-ensemble de variables sélectionnées de manière aléatoire à chaque nœud. 
            Ce processus favorise la diversité entre les arbres, ce qui réduit le risque de surapprentissage.
            """)
        
        with st.expander('Caractéristiques', True):
            st.markdown("## 2. Caractéristiques de Random Forests")

            st.markdown("""
            - Bagging (Bootstrap Aggregating) : création de multiples sous-échantillons de données avec remplacement.
            - Sélection aléatoire de variables : à chaque division d'un nœud, seul un sous-ensemble aléatoire de caractéristiques est testé.
            - Vote majoritaire (classification) ou moyenne (régression) pour agréger les prédictions.
            """)
        
        with st.expander('Mode de fonctionnement', True):
            st.markdown("## 3. Mode de fonctionnement de Random Forests")

            st.markdown("""
            1. Générer plusieurs sous-échantillons du jeu de données d'origine (bootstrap).
            2. Pour chaque sous-échantillon, construire un arbre de décision complet sans élagage.
            3. À chaque division de l'arbre, choisir aléatoirement un sous-ensemble de caractéristiques.
            4. Agréger les prédictions de tous les arbres :
                - Classification : vote majoritaire
                - Régression : moyenne des prédictions
            """)

            st.markdown("### 📊 Schémas Illustratifs")
            st.markdown("""
            Le schéma ci-dessous illustre une **forêt aléatoire** composée de plusieurs **arbres de décision**,
            dont les résultats sont combinés par **vote majoritaire** pour aboutir à une prédiction finale.
            """)

            # Affichage de l'image
            col1, col2 = st.columns(2)
            with col1:
                image0 = Image.open("assets/image0.png")
                small_image0 = image0.resize((600, 400))
                st.image(small_image0, caption="Schéma Visuel de Bagging")

            with col2:
                image = Image.open("assets/image.png")
                small_image = image.resize((600, 400))
                st.image(small_image, caption="Schéma Visuel de la Forêt Aléatoire")
        
        with st.expander('Avantages et Inconvénients', True):
            st.markdown("## 4. Avantages et Inconvénients")

            st.markdown("### Avantages")
            st.markdown("""
                - Réduction du surapprentissage grâce à l'agrégation.
                - Excellente précision pour des problèmes complexes.
                - Gère les valeurs manquantes et les variables catégorielles.
                - Capable de détecter l'importance des variables.
            """)
            
            st.markdown("### Inconvénients")
            st.markdown("""
                - Moins interprétable qu'un arbre de décision unique.
                - Plus coûteux en ressources (temps de calcul et mémoire).
                - Difficile à déployer sur des systèmes temps réel en cas de grande forêt.
            """)
        
        with st.expander('Domaines et champs d\'application', True):
            st.markdown("## 5. Domaines et champs d'application")

            st.markdown("""
                - Financial market prediction
                - Healthcare diagnostics
                - Image classification
                - Remote sensing
            """)

    with tab3:
        st.header("L’algorithme de Clustering K-Medoids")

        # Section 1
        with st.expander('Description', True):
            st.markdown("## 1. Description de l’algorithme K-Medoids")
            st.write("""
            L’algorithme K-Medoids est une méthode de clustering dans le cadre de l’apprentissage non supervisé. 
            Il permet de diviser un ensemble de données en k clusters. Contrairement à l’algorithme K-Means, 
            qui utilise des centroïdes comme centres des clusters, K-Medoids choisit des points réels du jeu 
            de données comme représentants des clusters, appelés **medoids**.

            Les medoids sont des points du jeu de données qui minimisent la distance totale à tous les autres 
            points du même cluster, ce qui les rend plus robustes aux valeurs aberrantes par rapport aux centroïdes.
            """)

        # Section 2
        with st.expander('Caractéristiques', True):
            st.markdown("## 2. Caractéristiques de K-Medoids")
            st.markdown("""
            - **Clustering basé sur les distances** : utilisation de mesures comme la distance euclidienne ou Manhattan.  
            - **Robustesse aux outliers** : plus résistant que K-Means.  
            - **Coût élevé en temps** : plus lent que K-Means, surtout pour de gros datasets.  
            - **Algorithme itératif** : convergence après plusieurs itérations.  
            - **Centres = objets réels** : contrairement à K-Means.
            """)

        # Section 3
        with st.expander('Mode de fonctionnement', True):
            st.markdown("## 3. Mode de fonctionnement de K-Medoids")

            st.markdown("### Étape 1 : Initialisation")
            st.write(" Choisir k points al´eatoires dans l’ensemble de donn´ees comme les m´ edino¨ıdes.")

            st.markdown("### Étape 2 : Assignation des points")
            st.write("Chaque point $x_i$ est affecté au medoid $m_j$ le plus proche :")
            st.latex(r"j = \arg\min_{1 \leq j \leq k} d(x_i, m_j)")
            st.write("où $d(x_i, m_j)$ est la distance euclidienne ou Manhattan.")

            st.markdown("### Étape 3 : Mise à jour (Swap)")
            st.write("Tester les échanges entre un medoid $m_j$ et un autre point $x_h$, recalculer :")
            st.latex(r"J = \sum_{j=1}^{k} \sum_{x_i \in C_j} d(x_i, m_j)")
            st.write("Si le coût diminue, accepter l’échange.")

            st.markdown("### Étape 4 : Convergence")
            st.write("Répéter jusqu’à stabilisation ou atteinte du nombre d’itérations.")

        # Section 4
        with st.expander('Avantages et Inconvénients', True):
            st.markdown("## 4. Avantages et Inconvénients")

            st.markdown("### Avantages")
            st.markdown("""
            - Résistant aux outliers.  
            - Adapté aux données non numériques.  
            - Interprétabilité des centres (objets réels).
            """)

            st.markdown("### Inconvénients")
            st.markdown("""
            - Complexité computationnelle élevée : $O(k(n-k)^2)$  
            - Sensible à l’initialisation.  
            - Moins adapté aux très grands ensembles de données.
            """)

        # Section 5
        with st.expander('Domaines et champs d\'application', True):
            st.markdown("## 5. Domaines et champs d'application")
            st.markdown("""
            - **Segmentation de marché**  
            - **Bioinformatique**  
            - **Analyse de texte**  
            - **Réseaux de capteurs**  
            - **Médecine**
            """)

