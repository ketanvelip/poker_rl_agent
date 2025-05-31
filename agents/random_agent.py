import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Agent that selects random actions."""
    
    def select_action(self, game_state, valid_actions, player):
        """
        Select a random action from the valid actions.
        
        Args:
            game_state (dict): The current state of the game
            valid_actions (list): List of valid actions
            player (Player): The player making the decision
            
        Returns:
            tuple: (action_type, amount)
        """
        action_type = random.choice(valid_actions)
        
        if action_type in ['bet', 'raise']:
            # Random bet between min and max
            min_bet = game_state['current_bet'] * 2
            max_bet = player.chips
            amount = random.randint(min_bet, max_bet) if max_bet > min_bet else min_bet
            return (action_type, amount)
        
        return (action_type, 0)  # amount is 0 for fold, check, call
