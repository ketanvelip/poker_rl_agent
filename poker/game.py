from .card import Deck, Card

class PokerGame:
    """
    A Texas Hold'em poker game.
    
    Attributes:
        players (list): List of players in the game
        deck (Deck): The deck of cards
        community_cards (list): The community cards on the board
        pot (int): The current pot size
        current_player_idx (int): Index of the current player
        dealer_idx (int): Index of the dealer
        small_blind (int): Small blind amount
        big_blind (int): Big blind amount
        current_bet (int): Current bet amount for this round
        betting_round (str): Current betting round ('preflop', 'flop', 'turn', 'river')
    """
    
    def __init__(self, players, small_blind=5, big_blind=10):
        """
        Initialize a poker game with players and blinds.
        
        Args:
            players (list): List of players in the game
            small_blind (int, optional): Small blind amount. Defaults to 5.
            big_blind (int, optional): Big blind amount. Defaults to 10.
        """
        if len(players) < 2:
            raise ValueError("Need at least 2 players")
            
        self.players = players
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_player_idx = 0
        self.dealer_idx = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_bet = 0
        self.betting_round = 'preflop'
        self.side_pots = []
    
    def start_hand(self):
        """Start a new hand of poker."""
        # Reset game state
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = 'preflop'
        self.side_pots = []
        
        # Reset player hands
        for player in self.players:
            player.clear_hand()
        
        # Rotate dealer
        self.dealer_idx = (self.dealer_idx + 1) % len(self.players)
        
        # Shuffle deck
        self.deck.reset()
        self.deck.shuffle()
        
        # Post blinds
        sb_idx = (self.dealer_idx + 1) % len(self.players)
        bb_idx = (self.dealer_idx + 2) % len(self.players)
        
        self.pot += self.players[sb_idx].bet(self.small_blind)
        self.pot += self.players[bb_idx].bet(self.big_blind)
        self.current_bet = self.big_blind
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                card = self.deck.deal()
                if card:
                    player.add_card(card)
        
        # Set starting player (after big blind)
        self.current_player_idx = (bb_idx + 1) % len(self.players)
    
    def current_player(self):
        """Get the current player."""
        return self.players[self.current_player_idx]
    
    def next_player(self):
        """Move to the next active player."""
        start_idx = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if self.players[self.current_player_idx].is_active or self.current_player_idx == start_idx:
                break
    
    def deal_community_cards(self, count=1):
        """
        Deal community cards.
        
        Args:
            count (int, optional): Number of cards to deal. Defaults to 1.
        """
        for _ in range(count):
            card = self.deck.deal()
            if card:
                self.community_cards.append(card)
    
    def process_action(self, action_type, amount=0):
        """
        Process a player's action.
        
        Args:
            action_type (str): Type of action ('fold', 'check', 'call', 'bet', 'raise')
            amount (int, optional): Bet amount if applicable. Defaults to 0.
        """
        player = self.current_player()
        
        if action_type == 'fold':
            player.fold()
        
        elif action_type == 'check':
            if self.current_bet > player.current_bet:
                raise ValueError("Cannot check when there's a bet")
            player.check()
        
        elif action_type == 'call':
            call_amount = min(self.current_bet - player.current_bet, player.chips)
            if call_amount > 0:
                self.pot += player.bet(call_amount)
        
        elif action_type in ['bet', 'raise']:
            if amount < self.current_bet * 2:
                amount = self.current_bet * 2  # Minimum raise
            
            self.pot += player.bet(amount)
            self.current_bet = player.current_bet
            
            # Reset has_acted for other players since there's a new bet
            for p in self.players:
                if p != player and p.is_active:
                    p.has_acted = False
        
        # Move to next player
        self.next_player()
    
    def is_betting_round_over(self):
        """
        Check if the current betting round is over.
        
        Returns:
            bool: True if the betting round is over
        """
        active_players = [p for p in self.players if p.is_active]
        
        # If only one player remains active, the betting round is over
        if len(active_players) <= 1:
            return True
        
        # Betting round is over when all active players have acted and matched the current bet
        all_players_acted = all(p.has_acted for p in active_players)
        all_bets_matched = all(p.current_bet == self.current_bet for p in active_players)
        
        return all_players_acted and all_bets_matched
    
    def next_betting_round(self):
        """Move to the next betting round."""
        # Check if only one player is active - if so, they win automatically
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            self.betting_round = 'showdown'  # Set to showdown to signal end of hand
            return [winner]
        
        # Advance to the next betting round
        if self.betting_round == 'preflop':
            self.betting_round = 'flop'
            self.deal_community_cards(3)
        
        elif self.betting_round == 'flop':
            self.betting_round = 'turn'
            self.deal_community_cards(1)
        
        elif self.betting_round == 'turn':
            self.betting_round = 'river'
            self.deal_community_cards(1)
        
        elif self.betting_round == 'river':
            # After river betting is complete, always go to showdown
            self.betting_round = 'showdown'
            return self.showdown()
        
        # If we reach here, we're not at showdown yet
        # Reset betting for new round
        self.current_bet = 0
        for player in self.players:
            player.has_acted = False
            player.current_bet = 0
        
        # Start with player after dealer
        self.current_player_idx = (self.dealer_idx + 1) % len(self.players)
        
        # Skip inactive players
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            # If we've gone full circle, break to avoid infinite loop
            if self.current_player_idx == (self.dealer_idx + 1) % len(self.players):
                break
        
        # If we've reached river and completed betting, force showdown
        if self.betting_round == 'river' and self.is_betting_round_over():
            self.betting_round = 'showdown'
            return self.showdown()
    
    def showdown(self):
        """
        Determine the winner(s) at showdown.
        
        Returns:
            list: List of winner(s)
        """
        # Simplified hand evaluation - to be expanded
        # In a real implementation, we would evaluate poker hands here
        active_players = [p for p in self.players if p.is_active]
        
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            return [winner]
        
        # Placeholder: just return the first active player as winner
        # This will be replaced with actual hand evaluation
        winner = active_players[0]
        winner.chips += self.pot
        return [winner]
    
    def get_state(self, for_player=None):
        """
        Get the current game state.
        
        Args:
            for_player (Player, optional): If provided, hide information not visible to this player.
            
        Returns:
            dict: The game state
        """
        state = {
            'pot': self.pot,
            'community_cards': self.community_cards.copy(),
            'current_bet': self.current_bet,
            'betting_round': self.betting_round,
            'players': []
        }
        
        for player in self.players:
            player_state = {
                'name': player.name,
                'chips': player.chips,
                'is_active': player.is_active,
                'current_bet': player.current_bet,
                'hand': []
            }
            
            # Only show cards if it's the requesting player or showdown
            if player == for_player or self.betting_round == 'showdown':
                player_state['hand'] = player.hand.copy()
            
            state['players'].append(player_state)
        
        return state
