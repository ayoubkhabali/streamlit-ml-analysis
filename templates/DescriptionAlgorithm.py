import streamlit as st

def app():
    # Sidebar menu
    # st.sidebar.title('Description Algorithm Selection')
    algorithm = st.sidebar.radio(
        'Choose an algorithm:',
        ['Decision Trees', 'Random Forests', 'K-Medoids']
    )

    # Main content area
    if algorithm == 'Decision Trees':
        st.header('Decision Trees')
        
        with st.expander('Description'):
            st.markdown("""
            Decision Trees are supervised learning algorithms that create a flowchart-like structure for decision making.
            They break down a dataset into smaller subsets while incrementally developing an associated decision tree.
            """)
        
        with st.expander('Characteristics'):
            st.markdown("""
            - Tree-like structure with nodes and branches
            - Root node, internal nodes, and leaf nodes
            - Binary or multi-way splits
            - Hierarchical decision making process
            """)
        
        with st.expander('Operation Mode'):
            st.markdown("""
            1. **Feature Selection**: Choose the best feature for splitting
            2. **Split Point Decision**: Determine optimal split threshold
            3. **Tree Growing**: Recursive splitting until stopping criteria
            4. **Pruning**: Optimize tree size to prevent overfitting
            """)
        
        with st.expander('Advantages and Disadvantages'):
            st.subheader('Advantages')
            st.markdown("""
            - Easy to understand and interpret
            - Requires little data preparation
            - Can handle numerical and categorical data
            - Handles non-linear relationships well
            """)
            
            st.subheader('Disadvantages')
            st.markdown("""
            - Can create overly complex trees
            - Can overfit the data
            - Unstable (small variations in data can result in different trees)
            - Biased toward dominant classes
            """)
        
        with st.expander('Application Domains'):
            st.markdown("""
            - Medical diagnosis
            - Customer churn prediction
            - Credit risk assessment
            - Plant or animal species classification
            """)

    elif algorithm == 'Random Forests':
        st.header('Random Forests')
        
        with st.expander('Description'):
            st.markdown("""
            Random Forests is an ensemble learning method that operates by constructing multiple decision trees
            during training and outputting the class that is the mode of the classes (classification) or mean
            prediction (regression) of the individual trees.
            """)
        
        with st.expander('Characteristics'):
            st.markdown("""
            - Ensemble of decision trees
            - Random feature selection
            - Bootstrap sampling (Bagging)
            - Parallel processing capability
            """)
        
        with st.expander('Operation Mode'):
            st.markdown("""
            1. **Bootstrap Sampling**: Create multiple datasets from original data
            2. **Random Feature Selection**: Select random subset of features
            3. **Tree Building**: Build decision trees on bootstrap samples
            4. **Aggregation**: Combine predictions using voting or averaging
            """)
        
        with st.expander('Advantages and Disadvantages'):
            st.subheader('Advantages')
            st.markdown("""
            - Higher accuracy than single decision trees
            - Good handling of overfitting
            - Can handle large datasets
            - Provides feature importance rankings
            """)
            
            st.subheader('Disadvantages')
            st.markdown("""
            - Less interpretable than single decision trees
            - Computationally intensive
            - Requires more memory
            - May overfit in some noisy classification/regression tasks
            """)
        
        with st.expander('Application Domains'):
            st.markdown("""
            - Financial market prediction
            - Healthcare diagnostics
            - Image classification
            - Remote sensing
            """)

    else:  # K-Medoids
        # Title
        st.header("L’algorithme de Clustering K-Medoids")

        # Section 1
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
        st.markdown("## 2. Caractéristiques de K-Medoids")
        st.markdown("""
        - **Clustering basé sur les distances** : utilisation de mesures comme la distance euclidienne ou Manhattan.  
        - **Robustesse aux outliers** : plus résistant que K-Means.  
        - **Coût élevé en temps** : plus lent que K-Means, surtout pour de gros datasets.  
        - **Algorithme itératif** : convergence après plusieurs itérations.  
        - **Centres = objets réels** : contrairement à K-Means.
        """)

        # Section 3
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
        st.markdown("## 5. Domaines et champs d'application")
        st.markdown("""
        - **Segmentation de marché**  
        - **Bioinformatique**  
        - **Analyse de texte**  
        - **Réseaux de capteurs**  
        - **Médecine**
        """)

        # Section 6
        st.markdown("## 6. Comparaison avec d'autres algorithmes")

        st.markdown("""
        | Critère               | K-Medoids          | K-Means            | DBSCAN                   |
        |-----------------------|--------------------|---------------------|--------------------------|
        | Centres des clusters  | Points réels       | Moyennes            | Aucun centre fixe        |
        | Robustesse outliers   | Haute              | Faible              | Très bonne               |
        | Type de données       | Mixtes, numériques | Numériques continues| Numériques, géospatiales |
        | Nb clusters           | À définir          | À définir           | Automatique              |
        | Forme des clusters    | Sphérique/autre    | Sphérique           | Formes arbitraires       |
        | Temps de calcul       | Moyen à élevé      | Faible              | Variable selon densité   |
        """, unsafe_allow_html=True)

        # st.header('K-Medoids')
        
        # with st.expander('1. Description', True):
        #     st.markdown("""
        #         L’algorithme K-Medoids est une m´ethode de clustering dans le cadre de
        #         l’apprentissage non supervis´ e. Il permet de diviser un ensemble de donn´ees
        #         en k clusters. Contrairement ` a l’algorithme K-Means, qui utilise des cen
        #         tro¨ ıdes comme centres des clusters, K-Medoids choisit des points r´ eels du
        #         jeu de donn´ees comme repr´esentants des clusters, appel´es medoids.
        #         Les medoids sont des points du jeu de donn´ees qui minimisent la dis
        #         tance totale ` a tous les autres points du mˆ eme cluster, ce qui les rend plus
        #         robustes aux valeurs aberrantes par rapport aux centro¨ ıdes utilis´es dans
        #         K-Means
        #     """)
        
        # with st.expander('2. Caract´ eristiques', True):
        #     st.markdown("""
        #         **• Clustering bas´ e sur les distances :** Utilisation de mesures de dis
        #         tance comme la distance euclidienne ou Manhattan pour regrouper les
        #         objets similaires.
        #         **• Robustesse aux outliers :** Le choix des m´edino¨ ıdes rend cet algo
        #         rithme plus r´esistant aux valeurs extrˆemes compar´ e ` a K-Means.
        #         **• Coˆut ´ elev´ e en temps :** L’algorithme est plus lent que K-Means,
        #         surtout pour des ensembles de donn´ees volumineux.
        #         **• Algorithme it´ eratif :** Les clusters sont affin´ es `a chaque it´eration
        #         jusqu’` a ce qu’ils convergent.
        #         **• Les centres sont des objets r´ eels :** Contrairement ` a K-Means, les
        #         centres de clusters sont des objets r´ eels pr´esents dans les donn´ ees.
        #     """)
        
        # with st.expander('3. Mode de fonctionnement', True):
        #     st.markdown("""
        #         L’algorithme K-Medoids fonctionne par it´ erations, en cherchant ` a min
        #         imiser une fonction de coˆ ut qui mesure la distance entre les points d’un
        #         cluster et leur medoid. Voici les ´etapes principales du processus, accom
        #         pagn´ ees des explications math´ematiques :
                
        #         **3.1 ´ Etape 1 : Initialisation**
        #         Choisir k points al´eatoires dans l’ensemble de donn´ees comme les m´ edino¨ıdes
        #         initiaux.
                        
        #         **3.2 ´ Etape 2 : Assignation des points aux clusters**
        #         Chaque point xi est affect´e au medoid le plus proche. Le cluster Cj d’un
        #         medoid mj contient tous les points xi qui minimisent la distance ` a ce medoid
        #         :
        #         j =arg min
        #         1≤j≤k 
        #         d(xi,mj)
        #         o` u d(xi, mj) est la distance entre le point xi et le medoid mj, mesur´ee par
        #         la distance euclidienne ou Manhattan.
                        
        #         **3.3 ´ Etape 3 : Mise `a jour (Swap)**
        #         Pour chaque cluster Cj, tester tous les ´echanges possibles entre un medoid
        #         mj et un autre point xh, et recalculer la fonction de coˆ ut J pour chaque
        #         ´ echange :
        #         k
        #         J =
        #         j=1xi∈Cj
        #         d(xi, mj)
        #         Si le swap diminue le coˆut total, l’´echange est accept´ e.
                        
        #         **3.4 ´ Etape 4 : Convergence**
        #         R´ ep´eter les ´ etapes jusqu’` a ce que les m´ edino¨ ıdes ne changent plus ou que le
        #         nombre maximal d’it´erations soit atteint.
        #     """)
        
        # with st.expander('4. Advantages and Disadvantages', True):
        #     st.subheader('4.1 Advantages')
        #     st.markdown("""
        #         **• R´esistance aux outliers :** Les m´edino¨ ıdes sont moins influenc´es par
        #         les valeurs aberrantes que les centro¨ ıdes.
        #         **• Adapt´e ` a des donn´ ees non num´ eriques :** Peut ˆetre utilis´ e avec des
        #         donn´ ees mixtes (num´eriques et cat´egorielles).
        #         **• Interpr´ etabilit´ e des centres :** Les m´edino¨ ıdes sont des objets r´eels,
        #         facilitant leur interpr´ etation dans un contexte appliqu´e.
        #     """)
            
        #     st.subheader('4.2 Inconv´enients')
        #     st.markdown("""
        #         **• Complexit´e computationnelle ´ elev´ ee :** L’algorithme est plus coˆuteux
        #         que K-Means, surtout pour de grands ensembles de donn´ees. La com
        #         plexit´ e par it´ eration est g´ en´eralement O(k(n − k)2).
        #         **• Sensibilit´e ` a l’initialisation :** Le choix des m´edino¨ ıdes initiaux in
        #         f
        #         luence les r´esultats.
        #         **• Moins adapt´ e aux grands ensembles de donn´ ees :** ` A cause de
        #         la complexit´e, l’algorithme devient difficilement scalable pour de tr`es
        #         grands ensembles.
        #     """)
        
        # with st.expander('5. Domaines et champs d’application', True):
        #     st.markdown("""
        #         K-Medoids est utilis´e dans plusieurs domaines o`u les donn´ ees peuvent ˆetre
        #         bruit´ ees ou non num´ eriques. Voici quelques exemples :
                        
        #         **• Segmentation de march´ e :** Identifier des groupes de clients ayant
        #         des comportements similaires.
        #         **• Bioinformatique :** Regrouper des g` enes similaires dans l’analyse de
        #         s´ equences g´en´ etiques.
        #         **• Analyse de texte :** Regrouper des documents similaires par leurs
        #         caract´ eristiques textuelles.
        #         **• R´eseaux de capteurs :** Regroupement de capteurs g´eographiquement
        #         proches pour la gestion des donn´ees.
        #         **• M´edecine :** Segmenter des patients en groupes en fonction de leurs
        #         symptˆ omes ou diagnostics.
        #     """)


