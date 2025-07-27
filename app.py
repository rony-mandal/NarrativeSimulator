import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from simulation.model import NarrativeModel
from processing.narrative_processor import process_narratives, load_narrative_data

def run_dashboard():
    st.title("Narrative Spread Simulation")
    data_source = st.radio("Data Source", ["Manual Input", "Preloaded Data"])
    if data_source == "Manual Input":
        narrative_input = st.text_area("Enter narratives (one per line):")
        narrative_texts = [text.strip() for text in narrative_input.split('\n') if text.strip()]
        narratives = process_narratives(narrative_texts) if narrative_texts else {}
    else:
        narratives = load_narrative_data()
    
    if not narratives:
        st.warning("Please enter at least one narrative or ensure preloaded data exists.")
        return
    num_agents = st.slider("Number of agents", 10, 1000, 100)
    steps = st.slider("Simulation steps", 1, 100, 20)

    if st.button("Run Simulation"):
        st.subheader("Narratives")
        for nid, narrative in narratives.items():
            st.write(f"Narrative {nid}: {narrative['text']}")
        
        model = NarrativeModel(num_agents, narratives)
        for step in range(steps):
            model.step()
        
        df = model.get_data_frame()
        believer_columns = [col for col in df.columns if 'narrative_' in col and '_believers' in col]
        fig_believers = px.line(df, x='step', y=believer_columns, title='Narrative Believers Over Time')
        st.plotly_chart(fig_believers)
        fig_sentiment = px.line(df, x='step', y='avg_sentiment', title='Average Sentiment Over Time')
        st.plotly_chart(fig_sentiment)
        
        if model.network_data:
            last_network = model.network_data[-1]
            node_x = [random.random() for _ in last_network['nodes']]
            node_y = [random.random() for _ in last_network['nodes']]
            edge_x = []
            edge_y = []
            for src, tgt in last_network['edges']:
                src_idx = next(i for i, (nid, _) in enumerate(last_network['nodes']) if nid == src)
                tgt_idx = next(i for i, (nid, _) in enumerate(last_network['nodes']) if nid == tgt)
                edge_x.extend([node_x[src_idx], node_x[tgt_idx], None])
                edge_y.extend([node_y[src_idx], node_y[tgt_idx], None])
            
            fig_network = go.Figure(data=[go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=0.5), hoverinfo='none'),
                                        go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                                   marker=dict(size=10, color=[{'Influencer': 'red', 'Regular': 'blue', 'Skeptic': 'green'}[t] for _, t in last_network['nodes']])),
                                        ],
                       layout=go.Layout(showlegend=False, hovermode='closest', title='Agent Network at Step ' + str(last_network['step'])))
            st.plotly_chart(fig_network)

import random

if __name__ == "__main__":
    run_dashboard()