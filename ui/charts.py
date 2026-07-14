import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def _get_transparent_layout():
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': dict(t=30, l=10, r=10, b=10)
    }

def create_risk_pie_chart(data: pd.DataFrame) -> go.Figure:
    risk_counts = data['Outcome'].value_counts()
    labels = ['Normal', 'High Risk (Diabetic)']
    values = [risk_counts[0], risk_counts[1]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5,
        marker_colors=['#22C55E', '#EF4444'],
        textinfo='label+percent'
    )])
    fig.update_layout(**_get_transparent_layout(), showlegend=False)
    return fig

def create_age_glucose_scatter(data: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        data, 
        x='Age', 
        y='Glucose', 
        color='Outcome',
        color_continuous_scale=['#22C55E', '#EF4444'],
        size='BMI',
        opacity=0.7,
        hover_data=['Insulin', 'BloodPressure']
    )
    fig.update_layout(**_get_transparent_layout(), coloraxis_showscale=False)
    return fig

def create_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    corr = data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1
    ))
    fig.update_layout(**_get_transparent_layout())
    return fig

def create_feature_importance_chart(feature_imp: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        feature_imp, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(**_get_transparent_layout(), coloraxis_showscale=False)
    return fig

def create_gauge_chart(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Glucose (mg/dL)"},
        gauge = {
            'axis': {'range': [None, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.5)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 100], 'color': 'rgba(34,197,94,0.3)'},
                {'range': [100, 125], 'color': 'rgba(245,158,11,0.3)'},
                {'range': [125, 300], 'color': 'rgba(239,68,68,0.3)'}
            ]
        }
    ))
    fig.update_layout(height=250, **_get_transparent_layout())
    return fig

# --- NEW CHARTS ---

def create_violin_plot(data: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.violin(data, y=feature, color="Outcome", box=True, points="all",
                    color_discrete_map={0: '#22C55E', 1: '#EF4444'})
    fig.update_layout(**_get_transparent_layout())
    return fig

def create_box_plot(data: pd.DataFrame, y_feature: str, x_feature: str) -> go.Figure:
    fig = px.box(data, x=x_feature, y=y_feature, color=x_feature,
                 color_discrete_map={0: '#22C55E', 1: '#EF4444'})
    fig.update_layout(**_get_transparent_layout())
    return fig

def create_radar_chart(data: pd.DataFrame) -> go.Figure:
    # Scale features 0-1 for radar chart
    features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    df_mean = data.groupby('Outcome')[features].mean().reset_index()
    
    fig = go.Figure()
    
    for i in range(len(df_mean)):
        outcome = df_mean['Outcome'].iloc[i]
        color = '#EF4444' if outcome == 1 else '#22C55E'
        name = 'High Risk' if outcome == 1 else 'Normal'
        
        # Min-Max scale just for visual
        vals = []
        for f in features:
            val = (df_mean[f].iloc[i] - data[f].min()) / (data[f].max() - data[f].min())
            vals.append(val)
            
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=features,
            fill='toself',
            name=name,
            line_color=color
        ))
        
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=True,
        **_get_transparent_layout()
    )
    return fig

def create_sunburst_chart(data: pd.DataFrame) -> go.Figure:
    # Create categorical bins for sunburst
    df = data.copy()
    df['Risk Category'] = df['Outcome'].map({0: 'Normal', 1: 'High Risk'})
    df['BMI Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Drop NAs
    df = df.dropna(subset=['BMI Category'])
    
    fig = px.sunburst(df, path=['Risk Category', 'BMI Category'], 
                      color='Risk Category', color_discrete_map={'Normal': '#22C55E', 'High Risk': '#EF4444'})
    fig.update_layout(**_get_transparent_layout())
    return fig

def create_treemap_chart(data: pd.DataFrame) -> go.Figure:
    df = data.copy()
    df['Risk Category'] = df['Outcome'].map({0: 'Normal', 1: 'High Risk'})
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young (<30)', 'Middle (30-50)', 'Senior (>50)'])
    
    df_count = df.groupby(['Risk Category', 'Age Group']).size().reset_index(name='Count')
    
    fig = px.treemap(df_count, path=[px.Constant("All Patients"), 'Risk Category', 'Age Group'], values='Count',
                     color='Risk Category', color_discrete_map={'(?)': 'gray', 'Normal': '#22C55E', 'High Risk': '#EF4444'})
    fig.update_layout(**_get_transparent_layout())
    return fig

def create_histogram(data: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.histogram(data, x=feature, color="Outcome", marginal="rug",
                       color_discrete_map={0: '#22C55E', 1: '#EF4444'})
    fig.update_layout(**_get_transparent_layout())
    return fig
