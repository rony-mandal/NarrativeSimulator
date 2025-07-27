import mesa

class NarrativeAgent(mesa.Agent):
    def __init__(self, model):  # Remove unique_id parameter - Mesa 3.x handles this automatically
        super().__init__(model)  # Only pass model to super().__init__()
        self.beliefs = {}  # {narrative_id: belief_score}
        self.sentiment = 0.0
        self.connections = []

    def step(self):
        for narrative_id, belief in self.beliefs.items():
            if belief > 0.5:
                for neighbor in self.connections:
                    neighbor.receive_narrative(narrative_id, belief)

    def receive_narrative(self, narrative_id, incoming_belief):
        if narrative_id not in self.beliefs:
            self.beliefs[narrative_id] = 0.0
        alpha = 0.3
        self.beliefs[narrative_id] = (1 - alpha) * self.beliefs[narrative_id] + alpha * incoming_belief
        narrative_sentiment = self.model.narratives[narrative_id]['sentiment']
        self.sentiment = (self.sentiment + narrative_sentiment) / 2