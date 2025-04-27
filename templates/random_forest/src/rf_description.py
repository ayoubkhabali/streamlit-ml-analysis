import streamlit as st

def show_rf_description():
    st.markdown(r"""
    # Algorithme de Forêt Aléatoire (Random Forest)
    
    ## Introduction
    
    La Forêt Aléatoire est une méthode d'apprentissage d'ensemble qui fonctionne en construisant plusieurs arbres de décision pendant l'entraînement et en produisant la classe qui est le mode des classes (classification) ou la prédiction moyenne (régression) des arbres individuels. Développée par Leo Breiman en 2001, elle répond à de nombreuses limitations des arbres de décision uniques tout en exploitant leurs forces.
    
    ## Fondement Mathématique
    
    ### Méthode d'Ensemble
    
    La Forêt Aléatoire combine les prédictions de plusieurs arbres de décision, où chaque arbre est entraîné sur un sous-ensemble aléatoire des données et des caractéristiques. La prédiction finale est donnée par:
    
    $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b$$
    
    où $B$ est le nombre d'arbres, et $\hat{y}_b$ est la prédiction du $b$-ème arbre.
    
    ### Bootstrap Aggregating (Bagging)
    
    Pour chaque arbre, un échantillon bootstrap $\mathcal{D}_b$ est créé à partir de l'ensemble de données original $\mathcal{D}$ par échantillonnage avec remplacement. Chaque échantillon bootstrap contient environ 63,2% des échantillons uniques de l'ensemble de données original. Ce pourcentage provient de la probabilité qu'un échantillon soit sélectionné au moins une fois: $1 - (1 - \frac{1}{N})^N \approx 1 - e^{-1} \approx 0,632$ pour un grand $N$.
    
    $$\mathcal{D}_b = \text{Bootstrap}(\mathcal{D})$$
    
    Les 36,8% d'échantillons restants non inclus dans l'échantillon bootstrap sont appelés échantillons "out-of-bag" (OOB), qui servent d'ensemble de validation intégré pour chaque arbre.
    
    ### Sélection Aléatoire des Caractéristiques
    
    À chaque nœud de chaque arbre de décision, un sous-ensemble aléatoire de caractéristiques est considéré pour la division:
    
    $$m = \sqrt{p} \text{ pour la classification, } m = \frac{p}{3} \text{ pour la régression}$$
    
    où $p$ est le nombre total de caractéristiques et $m$ est le nombre de caractéristiques considérées à chaque division. Cette randomisation des caractéristiques est cruciale pour réduire la corrélation entre les arbres, ce qui améliore la performance de l'ensemble.
    
    ### Critères de Division
    
    Pour les tâches de classification, les critères de division courants incluent:
    
    **Impureté de Gini**:
    $$\text{Gini}(t) = 1 - \sum_{i=1}^{C} p_i^2$$
    
    **Entropie**:
    $$\text{Entropie}(t) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
    
    où $p_i$ est la proportion de la classe $i$ au nœud $t$, et $C$ est le nombre de classes.
    
    Pour les tâches de régression, le critère est généralement:
    
    **Erreur Quadratique Moyenne (MSE)**:
    $$\text{MSE}(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$$
    
    où $N_t$ est le nombre d'échantillons au nœud $t$, $y_i$ est la valeur cible de l'échantillon $i$, et $\bar{y}_t$ est la valeur cible moyenne au nœud $t$.
    
    ## Étapes de l'Algorithme
    
    1. **Échantillonnage Bootstrap**: Pour $b = 1$ à $B$:
        - Tirer un échantillon bootstrap $\mathcal{D}_b$ de taille $N$ à partir des données d'entraînement.
    
    2. **Croissance de l'Arbre**: Pour chaque échantillon bootstrap $\mathcal{D}_b$, faire croître un arbre de décision non élagué $T_b$:
        - À chaque nœud, sélectionner aléatoirement $m$ caractéristiques parmi les $p$ caractéristiques.
        - Choisir la meilleure division parmi les $m$ caractéristiques selon la fonction objectif (par exemple, impureté de Gini, entropie).
        - Diviser le nœud en nœuds filles.
        - Répéter jusqu'à ce que les critères d'arrêt soient atteints (par exemple, taille minimale du nœud, profondeur maximale).
    
    3. **Prédiction**:
        - Pour la classification: $\hat{y} = \text{vote majoritaire}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B)$
        - Pour la régression: $\hat{y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b$
    
    ## Détails Techniques d'Implémentation
    
    ### Construction de l'Arbre
    
    Contrairement aux arbres de décision standard, les arbres de Forêt Aléatoire sont généralement développés jusqu'à leur profondeur maximale sans élagage. Cela est dû au fait que la nature d'ensemble de l'algorithme gère naturellement le surapprentissage qui se produirait dans des arbres individuels profonds.
    
    ### Parallélisation
    
    Comme chaque arbre est indépendant, l'entraînement de la Forêt Aléatoire peut être facilement parallélisé sur plusieurs cœurs ou machines, ce qui le rend efficace pour les grands ensembles de données.
    
    ### Besoins en Mémoire
    
    Les Forêts Aléatoires peuvent nécessiter une mémoire importante, surtout avec de nombreux arbres et des arbres profonds. La complexité mémoire est approximativement $O(B \cdot N \cdot \log(N))$ où $B$ est le nombre d'arbres et $N$ est le nombre d'échantillons.
    
    ### Complexité Temporelle
    
    La complexité temporelle pour l'entraînement est approximativement $O(B \cdot m \cdot N \cdot \log(N))$, où $m$ est le nombre de caractéristiques considérées à chaque division. Pour la prédiction, la complexité est $O(B \cdot \log(N))$.
    
    ## Importance des Caractéristiques
    
    Les Forêts Aléatoires fournissent une mesure de l'importance des caractéristiques basée sur la façon dont chaque caractéristique diminue l'impureté pondérée:
    
    $$\text{Importance}(X_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} p(t) \Delta i(t, j)$$
    
    où:
    - $p(t)$ est la proportion d'échantillons atteignant le nœud $t$
    - $\Delta i(t, j)$ est la diminution de l'impureté au nœud $t$ due à la caractéristique $j$
    
    Cette mesure peut être biaisée en faveur des caractéristiques à haute cardinalité. Une alternative est l'importance par permutation, qui mesure la diminution de la précision du modèle lorsque les valeurs d'une caractéristique sont permutées aléatoirement.
    
    ## Estimation de l'Erreur Out-of-Bag
    
    Comme chaque arbre est entraîné sur un échantillon bootstrap, environ 36,8% des données originales ne sont pas utilisées pour l'entraînement de chaque arbre (échantillons out-of-bag). Ces échantillons peuvent être utilisés pour estimer l'erreur de prédiction:
    
    $$\text{Erreur OOB} = \frac{1}{N} \sum_{i=1}^{N} I(\hat{y}_i^{OOB} \neq y_i)$$
    
    où $\hat{y}_i^{OOB}$ est la prédiction pour le $i$-ème échantillon en utilisant uniquement les arbres qui n'avaient pas cet échantillon dans leur échantillon bootstrap.
    
    L'erreur OOB fournit une estimation non biaisée de l'erreur de test et élimine le besoin d'un ensemble de validation séparé ou d'une validation croisée.
    
    ## Avantages
    
    - **Réduit le surapprentissage**: En moyennant plusieurs arbres, les Forêts Aléatoires réduisent le risque de surapprentissage par rapport aux arbres de décision individuels.
    - **Gère les données de haute dimension**: Efficace avec de nombreuses caractéristiques, même lorsque le nombre de caractéristiques dépasse le nombre d'observations.
    - **Importance des caractéristiques**: Fournit des méthodes intégrées pour évaluer l'importance des variables.
    - **Robustesse**: Résistant aux valeurs aberrantes et aux données non linéaires.
    - **Gestion des valeurs manquantes**: Peut gérer les valeurs manquantes grâce à des mesures de proximité ou des divisions de substitution.
    - **Pas de mise à l'échelle des caractéristiques requise**: Contrairement à de nombreux algorithmes, les Forêts Aléatoires ne nécessitent pas de normalisation des caractéristiques.
    - **Parallélisation**: L'entraînement peut être facilement parallélisé pour plus d'efficacité.
    - **Validation intégrée**: Les échantillons OOB fournissent un mécanisme de validation interne.
    - **Gère les données déséquilibrées**: Peut être pondéré pour gérer le déséquilibre des classes.
    - **Fonctionne pour la classification et la régression**: Polyvalent à travers les types de problèmes.
    
    ## Inconvénients
    
    - **Nature de boîte noire**: Moins interprétable qu'un seul arbre de décision.
    - **Ressources computationnelles**: Nécessite plus de puissance de calcul et de mémoire que des modèles plus simples.
    - **Temps d'entraînement**: Peut être lent à entraîner avec de nombreux arbres et de grands ensembles de données.
    - **Corrélation des caractéristiques**: Peut ne pas bien gérer les caractéristiques corrélées, car il pourrait surutiliser une caractéristique tout en ignorant les autres.
    - **Importance des caractéristiques biaisée**: Les mesures d'importance standard peuvent être biaisées en faveur des caractéristiques à haute cardinalité.
    - **Extrapolation limitée**: Comme toutes les méthodes basées sur les arbres, ne peut pas extrapoler au-delà de la plage des données d'entraînement.
    - **Possibilité de surapprentissage**: Avec des données bruyantes, peut toujours surapprendre malgré l'approche d'ensemble.
    - **Réglage des hyperparamètres**: La performance dépend du réglage approprié des hyperparamètres.
    - **Taille du modèle importante**: Les modèles résultants peuvent être très volumineux, rendant le déploiement difficile dans des environnements à ressources limitées.
    - **Pas optimal pour les relations linéaires**: D'autres algorithmes peuvent mieux performer lorsque les relations sont vraiment linéaires.
    
    ## Hyperparamètres
    
    - **Nombre d'arbres** ($B$): Plus d'arbres donnent généralement de meilleures performances mais augmentent le coût computationnel. Typiquement varie de 100 à plusieurs milliers.
    - **Profondeur maximale**: Contrôle la profondeur maximale de chaque arbre. Des arbres plus profonds capturent des modèles plus complexes mais peuvent conduire au surapprentissage.
    - **Échantillons minimums pour diviser**: Nombre minimum d'échantillons requis pour diviser un nœud. Des valeurs plus élevées empêchent la création de nœuds avec peu d'échantillons.
    - **Échantillons minimums par feuille**: Nombre minimum d'échantillons requis à un nœud feuille. Assure que les feuilles représentent un nombre minimum d'observations.
    - **Caractéristiques maximales** ($m$): Nombre de caractéristiques à considérer lors de la recherche de la meilleure division. Contrôle l'aléatoire dans la sélection des caractéristiques.
    - **Bootstrap**: Si l'on utilise des échantillons bootstrap. Le régler à False signifie utiliser l'ensemble de données entier pour chaque arbre.
    - **Poids des classes**: Poids associés aux classes pour traiter les ensembles de données déséquilibrés.
    
    ## Compromis Biais-Variance
    
    Les Forêts Aléatoires réduisent la variance en moyennant plusieurs arbres de décision, tout en maintenant le même biais que les arbres individuels. La randomisation dans la sélection des caractéristiques et l'échantillonnage bootstrap aide à décorréler les arbres, rendant l'ensemble plus robuste.
    
    $$\text{MSE} = \text{Biais}^2 + \text{Variance} + \text{Erreur Irréductible}$$
    
    Les Forêts Aléatoires réduisent efficacement le terme de variance dans cette équation. La décomposition biais-variance montre que l'erreur attendue d'un ensemble de $B$ arbres avec une corrélation $\rho$ entre deux arbres quelconques est:
    
    $$\text{Erreur} = \text{Biais}^2 + \rho \cdot \sigma^2 + \frac{1-\rho}{B} \cdot \sigma^2$$
    
    où $\sigma^2$ est la variance d'un arbre individuel. Lorsque $B$ augmente, le troisième terme approche zéro, laissant le biais et la variance corrélée. C'est pourquoi réduire la corrélation entre les arbres (par la randomisation des caractéristiques) est crucial pour la performance des Forêts Aléatoires.
    """)

# if __name__ == "__main__":
#     show_rf_description()