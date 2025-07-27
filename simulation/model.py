import mesa
from mesa import Model
import numpy as np
import pandas as pd
from .agents import NarrativeAgent

class NarrativeModel(Model):
    def __init__(self, num_agents, narratives):
        super().__init__()
        self.num_agents = num_agents
        self.narratives = narratives  # dict {id: {'text': ..., 'embedding': ..., 'sentiment': ...}}
        
        # Create agents - Mesa 3.x uses AgentSet instead of schedulers
        for i in range(num_agents):
            agent = NarrativeAgent(self)  # Only pass model, unique_id is automatic
            # Agents are automatically added to self.agents (AgentSet) when created with model parameter
        
        # Set up connections between agents
        for agent in self.agents:
            if len(self.agents) >= 5:
                # Convert AgentSet to list for numpy.random.choice
                agent_list = list(self.agents)
                agent.connections = np.random.choice(agent_list, size=5, replace=False).tolist()
            else:
                # If we have fewer than 5 agents, connect to all others
                agent.connections = [a for a in self.agents if a != agent]
        
        # Seed the first narrative if exists
        if narratives and self.agents:
            first_narrative_id = list(narratives.keys())[0]
            # Get first agent from AgentSet
            first_agent = list(self.agents)[0]
            first_agent.beliefs[first_narrative_id] = 1.0
        
        # Initialize data collection
        self.data = {'step': []}
        for nid in narratives:
            self.data[f'narrative_{nid}_believers'] = []
        self.data['avg_sentiment'] = []

    def step(self):
        """Execute one step of the model"""
        # In Mesa 3.x, use AgentSet.do() to execute all agents
        self.agents.do("step")
        
        # Increment step counter manually (Mesa 3.x doesn't have built-in step counter)
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        # Collect data for this step
        step_data = {'step': self._step_count}
        
        # Count believers for each narrative
        for nid in self.narratives:
            believers = sum(1 for agent in self.agents 
                          if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            step_data[f'narrative_{nid}_believers'] = believers
        
        # Calculate average sentiment
        if self.agents:
            step_data['avg_sentiment'] = np.mean([agent.sentiment for agent in self.agents])
        else:
            step_data['avg_sentiment'] = 0.0
        
        # Store the data
        for key, value in step_data.items():
            self.data[key].append(value)

    def run_model(self, step_count=100):
        """Run the model for a given number of steps"""
        for i in range(step_count):
            self.step()

    def get_data_frame(self):
        """Return the collected data as a pandas DataFrame"""
        return pd.DataFrame(self.data)

    def reset(self):
        """Reset the model to initial state"""
        # Clear existing agents
        for agent in list(self.agents):
            agent.remove()
        
        # Reset step counter
        self._step_count = 0
        
        # Reinitialize
        self.__init__(self.num_agents, self.narratives)