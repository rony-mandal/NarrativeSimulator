import mesa
from mesa import Model
import numpy as np
import pandas as pd
from .agents import NarrativeAgent

class NarrativeModel(Model):
    def __init__(self, num_agents, initial_narratives):
        super().__init__()
        self.num_agents = num_agents
        self.narratives = initial_narratives.copy()  # Initial narratives
        self.counter_narratives = {}  # Dynamically generated counter-narratives
        
        # Create agents
        for i in range(num_agents):
            agent = NarrativeAgent(self)
        
        # Set up connections
        for agent in self.agents:
            if len(self.agents) >= 5:
                agent_list = list(self.agents)
                agent.connections = np.random.choice(agent_list, size=5, replace=False).tolist()
            else:
                agent.connections = [a for a in self.agents if a != agent]
        
        # Seed the first narrative
        if initial_narratives and self.agents:
            first_narrative_id = list(initial_narratives.keys())[0]
            first_agent = list(self.agents)[0]
            first_agent.beliefs[first_narrative_id] = 1.0
        
        # Initialize data collection
        self.data = {'step': []}
        for nid in initial_narratives:
            self.data[f'narrative_{nid}_believers'] = []
        self.data['avg_sentiment'] = []
        self.network_data = [] 
        self._step_count = 0

    def step(self):
        self.agents.do("step")
        self._step_count += 1
        
        # Generate adaptive counter-narrative if dominant narrative exists
        if self._step_count % 5 == 0 and self.narratives:
            dominant_nid = max(self.narratives, key=lambda x: sum(1 for a in self.agents if x in a.beliefs and a.beliefs[x] > 0.5))
            counter_text = f"No, {self.narratives[dominant_nid]['text'].lower().replace('is', 'is not')}"
            if counter_text not in [n['text'] for n in self.narratives.values()]:
                counter_nid = max(self.narratives.keys()) + 1
                self.counter_narratives[counter_nid] = {
                    'text': counter_text,
                    'embedding': self.narratives[dominant_nid]['embedding'],  # Placeholder, refine with model
                    'sentiment': -self.narratives[dominant_nid]['sentiment']
                }
                self.narratives[counter_nid] = self.counter_narratives[counter_nid]
                
                # Initialize data for new counter-narrative with zeros for previous steps
                self.data[f'narrative_{counter_nid}_believers'] = [0] * (self._step_count - 1)
                
                # Seed counter-narrative in a random agent
                agents_list = list(self.agents)
                if agents_list:
                    agents_list[0].beliefs[counter_nid] = 1.0
        
        # Collect data for this step
        step_data = {'step': self._step_count}
        
        # Collect data for all narratives (initial + counter)
        all_narratives = {**self.narratives, **self.counter_narratives}
        for nid in all_narratives:
            believers = sum(1 for agent in self.agents if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            step_data[f'narrative_{nid}_believers'] = believers
        
        if self.agents:
            step_data['avg_sentiment'] = np.mean([agent.sentiment for agent in self.agents])
        else:
            step_data['avg_sentiment'] = 0.0
        
        # Append data to ensure all arrays have same length
        for key, value in step_data.items():
            if key in self.data:
                self.data[key].append(value)
        
        # Collect network data
        self.network_data.append({
            'step': self._step_count,
            'nodes': [(a.unique_id, a.type) for a in self.agents],
            'edges': [(a.unique_id, n.unique_id) for a in self.agents for n in a.connections]
        })

    def get_data_frame(self):
        # Ensure all arrays have the same length before creating DataFrame
        max_length = len(self.data['step'])
        for key in self.data:
            if len(self.data[key]) < max_length:
                # Pad with zeros for missing steps
                self.data[key].extend([0] * (max_length - len(self.data[key])))
        
        return pd.DataFrame(self.data)

    def get_network_data(self):
        return self.network_data