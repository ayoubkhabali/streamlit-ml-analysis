import streamlit as st
from templates.random_forest.src.model import RandomForestModel
from templates.random_forest.src.preprocessing import DataPreprocessor
from templates.random_forest.src.visualization import Visualizer
import pandas as pd

st.set_page_config(page_title="Random Forest Analysis", layout="wide")

def main():
    st.markdown(
        "<h1 style='text-align: center; color: #4F8BF9;'>Random Forest Analysis Dashboard</h1>",
        unsafe_allow_html=True
    )

    # Parameters in an expander at the top
    with st.expander("🔧 Model Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider('Number of trees', 10, 200, 100)
        with col2:
            max_depth = st.slider('Maximum depth', 1, 30, 10)
        with col3:
            min_samples_split = st.slider('Minimum samples to split', 2, 10, 2)
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }

    st.markdown("---")

    # Data upload section
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Data preview in a nice box
        st.markdown("#### 👀 Dataset Preview")
        st.dataframe(data.head(), use_container_width=True)

        # Feature selection in columns
        st.markdown("### 🏷️ Feature Selection")
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox("Select target column", data.columns)
        with col2:
            feature_columns = st.multiselect(
                "Select feature columns",
                [col for col in data.columns if col != target_column]
            )

        if feature_columns and target_column:
            st.markdown("---")
            st.markdown("### 🚀 Train and Evaluate Model")
            train_btn = st.button("Train Model", use_container_width=True)
            if train_btn:
                progress_bar = st.progress(0)
                preprocessor = DataPreprocessor()
                visualizer = Visualizer()

                # Preprocess data
                X, y = preprocessor.prepare_data(data, feature_columns, target_column)
                progress_bar.progress(25)

                # Initialize and train model
                model = RandomForestModel(params)
                model.train(X, y)
                progress_bar.progress(50)

                # Display results in columns
                st.markdown("#### 📊 Model Performance")
                metrics = model.get_metrics()
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                mcol2.metric("Precision", f"{metrics['precision']:.2%}")
                mcol3.metric("Recall", f"{metrics['recall']:.2%}")
                mcol4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                
                progress_bar.progress(75)

                # Visualizations in expanders for a clean look
                with st.expander("📊 Performance Metrics Visualization", expanded=True):
                    visualizer.plot_performance_metrics(metrics)
                    
                with st.expander("🔎 Feature Importance"):
                    visualizer.plot_feature_importance(model, feature_columns)
                with st.expander("🧮 Confusion Matrix"):
                    visualizer.plot_confusion_matrix(model)
                with st.expander("📋 Prediction Results"):
                    visualizer.plot_predictions(model)

                progress_bar.progress(100)
                st.success("Model training completed!")

if __name__ == "__main__":
    main()