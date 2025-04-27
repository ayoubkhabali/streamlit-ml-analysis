import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    def plot_feature_importance(self, model, feature_names):
        importance = model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        st.write("### Feature Importance")
        fig = px.bar(feature_importance, x='importance', y='feature',
                    orientation='h', title='Feature Importance')
        st.plotly_chart(fig)
        
    def plot_confusion_matrix(self, model):
        cm = model.get_confusion_matrix()
        st.write("### Confusion Matrix")
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       title="Confusion Matrix")
        st.plotly_chart(fig)
        
    def plot_predictions(self, model):
        st.write("### Prediction Results")
        results = pd.DataFrame({
            'Actual': model.y_test,
            'Predicted': model.predictions
        })
        st.write(results)
        
    def plot_performance_metrics(self, metrics):
        st.write("### Performance Metrics Visualization")
        
        # Create a radar chart for the metrics
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1_score']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Model Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Model Performance Metrics"
        )
        
        st.plotly_chart(fig)
        
        # Also display as a bar chart for comparison
        metrics_df = pd.DataFrame({
            'Metric': categories,
            'Value': values
        })
        
        bar_fig = px.bar(metrics_df, x='Metric', y='Value', 
                         title='Performance Metrics Comparison',
                         color='Metric', text_auto='.2%')
        bar_fig.update_layout(yaxis_range=[0, 1])
        
        st.plotly_chart(bar_fig)