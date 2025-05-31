class Player:
    """
    A poker player with cards, chips, and actions.
    
    Attributes:
        name (str): The player's name
        chips (int): The player's chip stack
        hand (list): The player's hole cards
        is_active (bool): Whether the player is still in the current hand
        has_acted (bool): Whether the player has acted in the current betting round
    """
    
    def __init__(self, name, chips=1000):
        """
        Initialize a player with a name and chips.
        
        Args:
            name (str): The player's name
            chips (int, optional): Starting chip count. Defaults to 1000.
        """
        self.name = name
        self.chips = chips
        self.hand = []
        self.is_active = True
        self.has_acted = False
        self.current_bet = 0
    
    def add_card(self, card):
        """
        Add a card to the player's hand.
        
        Args:
            card (Card): The card to add
        """
        self.hand.append(card)
    
    def clear_hand(self):
        """Clear the player's hand and reset state for a new round."""
        self.hand = []
        self.is_active = True
        self.has_acted = False
        self.current_bet = 0
    
    def bet(self, amount):
        """
        Place a bet.
        
        Args:
            amount (int): The amount to bet
            
        Returns:
            int: The actual amount bet
        
        Raises:
            ValueError: If the bet amount is invalid
        """
        if amount <= 0:
            raise ValueError("Bet amount must be positive")
        if amount > self.chips:
            amount = self.chips  # All-in
        
        self.chips -= amount
        self.current_bet += amount
        self.has_acted = True
        return amount
    
    def fold(self):
        """Fold the hand."""
        self.is_active = False
        self.has_acted = True
    
    def check(self):
        """Check (bet nothing)."""
        self.has_acted = True
    
    def __str__(self):
        """Return string representation of the player."""
        return f"{self.name} (${self.chips})"


class HumanPlayer(Player):
    """A human player that makes decisions through user input."""
    
    def act(self, game_state, valid_actions):
        """
        Get action from human input.
        
        Args:
            game_state (dict): The current state of the game
            valid_actions (list): List of valid actions the player can take
            
        Returns:
            tuple: (action_type, amount)
        """
        # This will be implemented in the UI layer
        # For now, we'll just return a placeholder
        return ("check", 0)


class AIPlayer(Player):
    """An AI player that makes decisions based on an agent."""
    
    def __init__(self, name, agent, chips=1000):
        """
        Initialize an AI player with a name, agent, and chips.
        
        Args:
            name (str): The player's name
            agent: The agent that makes decisions
            chips (int, optional): Starting chip count. Defaults to 1000.
        """
        super().__init__(name, chips)
        self.agent = agent
    
    def act(self, game_state, valid_actions):
        """
        Get action from the agent.
        
        Args:
            game_state (dict): The current state of the game
            valid_actions (list): List of valid actions the player can take
            
        Returns:
            tuple: (action_type, amount)
        """
        return self.agent.select_action(game_state, valid_actions, self)
