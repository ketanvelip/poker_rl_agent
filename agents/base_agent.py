class BaseAgent:
    """Base class for all poker agents."""
    
    def select_action(self, game_state, valid_actions, player):
        """
        Select an action based on the current game state.
        
        Args:
            game_state (dict): The current state of the game
            valid_actions (list): List of valid actions
            player (Player): The player making the decision
            
        Returns:
            tuple: (action_type, amount)
        """
        raise NotImplementedError("Subclasses must implement select_action")
