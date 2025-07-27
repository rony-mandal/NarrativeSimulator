import streamlit as st
import plotly.express as px
import pandas as pd
from simulation.model import NarrativeModel
from processing.narrative_processor import process_narratives

def run_dashboard():
    st.title("Narrative Spread Simulation")
    narrative_input = st.text_area("Enter narratives (one per line):")
    narrative_texts = [text.strip() for text in narrative_input.split('\n') if text.strip()]
    if not narrative_texts:
        st.warning("Please enter at least one narrative.")
        return
    num_agents = st.slider("Number of agents", 10, 1000, 100)
    steps = st.slider("Simulation steps", 1, 100, 20)

    if st.button("Run Simulation"):
        narratives = process_narratives(narrative_texts)
        st.subheader("Narratives")
        for nid, narrative in narratives.items():
            st.write(f"Narrative {nid}: {narrative['text']}")
        
        model = NarrativeModel(num_agents, narratives)
        for step in range(steps):
            if step == steps // 2 and len(narratives) > 1:
                # Introduce counter-narrative (assuming narrative 1 is the counter)
                model.agents[10].beliefs[1] = 1.0
            model.step()
        
        df = pd.DataFrame(model.data)
        believer_columns = [f'narrative_{nid}_believers' for nid in narratives]
        fig_believers = px.line(df, x='step', y=believer_columns, title='Narrative Believers Over Time')
        st.plotly_chart(fig_believers)
        fig_sentiment = px.line(df, x='step', y='avg_sentiment', title='Average Sentiment Over Time')
        st.plotly_chart(fig_sentiment)

if __name__ == "__main__":
    run_dashboard()