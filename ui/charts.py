import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_gauge_chart(predicted_glucose: float, risk_color: str) -> go.Figure:
    """
    Creates a gauge chart for predicted glucose level.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_glucose,
        title={'text': "Glucose Saturation Level", 'font': {'size': 16, 'color': '#64748B', 'family': 'Inter'}},
        number={'font': {'size': 40, 'color': '#0F172A', 'family': 'Poppins'}},
        gauge={
            'axis': {'range': [0, 250], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
            'bar': {'color': risk_color.replace('var(--danger)', '#EF4444').replace('var(--accent)', '#22C55E')},
            'bgcolor': 'rgba(255,255,255,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': 'rgba(34, 197, 94, 0.1)'},
                {'range': [100, 125], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [125, 250], 'color': 'rgba(239, 68, 68, 0.1)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_risk_pie_chart(data: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing the distribution of diabetes risk in the dataset.
    """
    fig = px.pie(data, names='Outcome', title="", hole=0.7,
                     color_discrete_sequence=['#2563EB', '#EF4444'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
    fig.add_annotation(text="Risk", x=0.5, y=0.5, font_size=20, showarrow=False)
    return fig

def create_age_glucose_scatter(data: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Age vs Glucose colored by Outcome.
    """
    fig = px.scatter(data, x='Age', y='Glucose', color='Outcome',
                              color_discrete_map={0: '#14B8A6', 1: '#F59E0B'},
                              opacity=0.6)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               margin=dict(t=0, b=0, l=0, r=0), height=300)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    return fig

def create_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """
    Heatmap of numerical feature correlations.
    """
    fig = px.imshow(data.corr(numeric_only=True), 
                      color_continuous_scale='Blues',
                      aspect="auto")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

def create_feature_importance_chart(feature_imp: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing feature importance.
    """
    fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Teal')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=0, b=0, l=0, r=0), height=350)
    fig.update_yaxes(categoryorder="total ascending")
    return fig
