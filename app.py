import streamlit as st

st.set_page_config(page_title="ML Algorithms Analysis", layout="wide")

st.title('Machine Learning Algorithms Analysis')

# Sidebar menu
st.sidebar.title('Algorithm Selection')
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
    st.header('K-Medoids')
    
    with st.expander('Description'):
        st.markdown("""
        K-Medoids is a clustering algorithm related to k-means clustering. It's a partitioning technique
        that clusters the dataset of n objects into k clusters, where k is a predefined number.
        Unlike k-means, k-medoids uses actual data points as cluster centers (medoids).
        """)
    
    with st.expander('Characteristics'):
        st.markdown("""
        - Partitional clustering algorithm
        - Uses actual data points as cluster centers
        - More robust to noise and outliers than k-means
        - Based on minimizing sum of dissimilarities
        """)
    
    with st.expander('Operation Mode'):
        st.markdown("""
        1. **Initialization**: Randomly select k data points as initial medoids
        2. **Assignment**: Assign each data point to nearest medoid
        3. **Update**: Find new medoid that minimizes total distance
        4. **Iteration**: Repeat steps 2-3 until convergence
        """)
    
    with st.expander('Advantages and Disadvantages'):
        st.subheader('Advantages')
        st.markdown("""
        - More robust to noise and outliers
        - No need to specify mean or median
        - Works with arbitrary distance metrics
        - Better interpretability of cluster centers
        """)
        
        st.subheader('Disadvantages')
        st.markdown("""
        - Computationally more expensive than k-means
        - Needs to specify k beforehand
        - May converge to local optima
        - Sensitive to initialization
        """)
    
    with st.expander('Application Domains'):
        st.markdown("""
        - Customer segmentation
        - Document clustering
        - Spatial data analysis
        - Pattern recognition in bioinformatics
        """)
