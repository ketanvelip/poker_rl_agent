class Card:
    """
    A playing card with a suit and rank.
    
    Attributes:
        suit (str): The suit of the card ('hearts', 'diamonds', 'clubs', 'spades')
        rank (int): The rank of the card (2-14, where 14 is Ace)
        visible (bool): Whether the card is visible to all players
    """
    
    SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
    RANKS = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
             10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    
    def __init__(self, suit, rank, visible=True):
        """
        Initialize a card with a suit and rank.
        
        Args:
            suit (str): The suit of the card
            rank (int): The rank of the card
            visible (bool, optional): Whether the card is visible. Defaults to True.
        """
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
            
        self.suit = suit
        self.rank = rank
        self.visible = visible
    
    def __str__(self):
        """Return string representation of card."""
        if not self.visible:
            return "??"
        return f"{self.RANKS[self.rank]}{self.suit[0].upper()}"
    
    def __repr__(self):
        """Return string representation of card for debugging."""
        return self.__str__()
    
    def __eq__(self, other):
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __lt__(self, other):
        """Compare cards by rank."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank < other.rank


class Deck:
    """
    A deck of playing cards.
    
    Attributes:
        cards (list): The cards in the deck
    """
    
    def __init__(self):
        """Initialize a standard deck of 52 cards."""
        self.cards = []
        self.reset()
    
    def reset(self):
        """Reset the deck to a standard 52-card deck."""
        self.cards = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                self.cards.append(Card(suit, rank))
    
    def shuffle(self):
        """Shuffle the deck."""
        import random
        random.shuffle(self.cards)
    
    def deal(self, visible=True):
        """
        Deal a card from the deck.
        
        Args:
            visible (bool, optional): Whether the card is visible. Defaults to True.
            
        Returns:
            Card: The dealt card, or None if the deck is empty
        """
        if not self.cards:
            return None
        
        card = self.cards.pop()
        card.visible = visible
        return card
    
    def __len__(self):
        """Return the number of cards in the deck."""
        return len(self.cards)
    
    def __str__(self):
        """Return string representation of the deck."""
        return f"Deck with {len(self.cards)} cards"
