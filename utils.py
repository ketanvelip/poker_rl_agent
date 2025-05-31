import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_hand(hand, community_cards):
    """
    Evaluate a poker hand to determine its rank.
    This is a simplified implementation for demonstration purposes.
    
    Args:
        hand (list): List of Card objects in the player's hand
        community_cards (list): List of Card objects on the board
        
    Returns:
        tuple: (hand_rank, tiebreaker_values)
    """
    # Combine hand and community cards
    all_cards = hand + community_cards
    
    # Check for different hand combinations
    # 9: Royal Flush, 8: Straight Flush, 7: Four of a Kind, 6: Full House,
    # 5: Flush, 4: Straight, 3: Three of a Kind, 2: Two Pair, 1: Pair, 0: High Card
    
    # Count cards by rank
    rank_counts = {}
    for card in all_cards:
        if card.rank in rank_counts:
            rank_counts[card.rank] += 1
        else:
            rank_counts[card.rank] = 1
    
    # Count cards by suit
    suit_counts = {}
    for card in all_cards:
        if card.suit in suit_counts:
            suit_counts[card.suit] += 1
        else:
            suit_counts[card.suit] = 1
    
    # Check for flush
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    # Check for straight
    ranks = sorted([card.rank for card in all_cards])
    unique_ranks = sorted(list(set(ranks)))
    straight = False
    straight_high = None
    
    # Special case: A-5 straight
    if 14 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 4 in unique_ranks and 5 in unique_ranks:
        straight = True
        straight_high = 5
    
    # Normal straights
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i:i+5] == list(range(unique_ranks[i], unique_ranks[i] + 5)):
            straight = True
            straight_high = unique_ranks[i+4]
    
    # Evaluate hand
    # Four of a kind
    for rank, count in rank_counts.items():
        if count == 4:
            return (7, [rank])
    
    # Full house
    three_of_a_kind = None
    pair = None
    for rank, count in rank_counts.items():
        if count == 3:
            three_of_a_kind = rank
        elif count == 2:
            pair = rank
    
    if three_of_a_kind is not None and pair is not None:
        return (6, [three_of_a_kind, pair])
    
    # Flush
    if flush_suit:
        flush_cards = [card for card in all_cards if card.suit == flush_suit]
        flush_ranks = sorted([card.rank for card in flush_cards], reverse=True)
        return (5, flush_ranks[:5])
    
    # Straight
    if straight:
        return (4, [straight_high])
    
    # Three of a kind
    if three_of_a_kind is not None:
        return (3, [three_of_a_kind])
    
    # Two pair
    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    if len(pairs) >= 2:
        return (2, sorted(pairs, reverse=True)[:2])
    
    # Pair
    if len(pairs) == 1:
        return (1, pairs)
    
    # High card
    high_cards = sorted(ranks, reverse=True)
    return (0, high_cards[:5])

def compare_hands(hand1, hand1_community, hand2, hand2_community):
    """
    Compare two poker hands to determine the winner.
    
    Args:
        hand1 (list): List of Card objects in player 1's hand
        hand1_community (list): List of Card objects on the board for player 1
        hand2 (list): List of Card objects in player 2's hand
        hand2_community (list): List of Card objects on the board for player 2
        
    Returns:
        int: 1 if hand1 wins, 2 if hand2 wins, 0 if tie
    """
    rank1, tiebreaker1 = evaluate_hand(hand1, hand1_community)
    rank2, tiebreaker2 = evaluate_hand(hand2, hand2_community)
    
    if rank1 > rank2:
        return 1
    elif rank2 > rank1:
        return 2
    
    # If ranks are equal, compare tiebreakers
    for tb1, tb2 in zip(tiebreaker1, tiebreaker2):
        if tb1 > tb2:
            return 1
        elif tb2 > tb1:
            return 2
    
    return 0  # Tie

def plot_training_progress(episodes, rewards, losses=None, window_size=10):
    """
    Plot training progress.
    
    Args:
        episodes (list): List of episode numbers
        rewards (list): List of rewards per episode
        losses (list, optional): List of losses per episode
        window_size (int, optional): Window size for moving average. Defaults to 10.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2 if losses else 1, 1)
    plt.plot(episodes, rewards, alpha=0.3, label='Raw')
    
    # Calculate and plot moving average
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean().values
        plt.plot(episodes[window_size-1:], moving_avg[window_size-1:], label=f'Moving Avg ({window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses:
        plt.subplot(1, 2, 2)
        plt.plot(episodes, losses, alpha=0.3, label='Raw')
        
        # Calculate and plot moving average
        if len(losses) >= window_size:
            moving_avg = pd.Series(losses).rolling(window=window_size).mean().values
            plt.plot(episodes[window_size-1:], moving_avg[window_size-1:], label=f'Moving Avg ({window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_pot_odds(pot_size, call_amount):
    """
    Calculate pot odds.
    
    Args:
        pot_size (int): Current pot size
        call_amount (int): Amount to call
        
    Returns:
        float: Pot odds as a percentage
    """
    return call_amount / (pot_size + call_amount) * 100

def calculate_hand_strength(hand, community_cards, num_opponents=2, num_simulations=1000):
    """
    Calculate the strength of a hand through Monte Carlo simulation.
    This is a simplified implementation for demonstration purposes.
    
    Args:
        hand (list): List of Card objects in the player's hand
        community_cards (list): List of Card objects on the board
        num_opponents (int, optional): Number of opponents. Defaults to 2.
        num_simulations (int, optional): Number of simulations. Defaults to 1000.
        
    Returns:
        float: Estimated probability of winning
    """
    from .poker.card import Deck, Card
    
    wins = 0
    
    for _ in range(num_simulations):
        # Create a new deck excluding known cards
        deck = Deck()
        deck.cards = [card for card in deck.cards if card not in hand and card not in community_cards]
        deck.shuffle()
        
        # Complete the community cards if needed
        sim_community = community_cards.copy()
        while len(sim_community) < 5:
            sim_community.append(deck.deal())
        
        # Deal opponent hands
        opponent_hands = []
        for _ in range(num_opponents):
            opp_hand = [deck.deal() for _ in range(2)]
            opponent_hands.append(opp_hand)
        
        # Compare hands
        player_wins = True
        for opp_hand in opponent_hands:
            result = compare_hands(hand, sim_community, opp_hand, sim_community)
            if result != 1:  # If player doesn't win
                player_wins = False
                break
        
        if player_wins:
            wins += 1
    
    return wins / num_simulations
