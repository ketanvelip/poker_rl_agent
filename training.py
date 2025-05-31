import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import argparse
import logging
from collections import deque
import pandas as pd
import seaborn as sns
from datetime import datetime

from poker.game import PokerGame
from poker.player import AIPlayer
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from utils import plot_training_progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("poker_rl")

# Action space definition
ACTION_SPACE = ['fold', 'check', 'call', 'bet', 'raise']

def train_agent(num_episodes=1000, save_freq=100, eval_freq=50, 
                learning_rate=0.001, memory_size=10000, batch_size=32,
                gamma=0.99, epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.995,
                target_update=10, alpha=0.6, beta_start=0.4, beta_end=1.0,
                n_steps=3, early_stopping=True, patience=100, min_epsilon=0.05,
                log_dir="logs", model_dir="data"):
    """
    Train a DQN agent to play poker with advanced reinforcement learning techniques.
    
    Args:
        num_episodes (int): Number of episodes to train for
        save_freq (int): Frequency to save the model
        eval_freq (int): Frequency to evaluate the model
        learning_rate (float): Learning rate for the optimizer
        memory_size (int): Size of the replay buffer
        batch_size (int): Batch size for training
        gamma (float): Discount factor
        epsilon_start (float): Starting exploration rate
        epsilon_end (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for epsilon
        target_update (int): Frequency to update target network
        alpha (float): Prioritization exponent for PER
        beta_start (float): Initial importance sampling correction
        beta_end (float): Final importance sampling correction
        n_steps (int): Number of steps for n-step returns
        early_stopping (bool): Whether to use early stopping
        patience (int): Number of evaluations without improvement before stopping
        min_epsilon (float): Minimum exploration rate
        log_dir (str): Directory to save logs
        model_dir (str): Directory to save models
    
    Returns:
        DQNAgent: The trained agent
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Define state dimension (simplified for now)
    state_dim = 50  # This should match the size of the state vector from DQNAgent._preprocess_state
    
    # Create the enhanced DQN agent
    dqn_agent = DQNAgent(
        input_dim=state_dim, 
        action_space=ACTION_SPACE,
        epsilon=epsilon_start,
        gamma=gamma,
        learning_rate=learning_rate,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update=target_update,
        alpha=alpha,
        beta=beta_start,
        n_steps=n_steps
    )
    
    # Create random opponents
    random_agent1 = RandomAgent()
    random_agent2 = RandomAgent()
    
    # Create players
    dqn_player = AIPlayer("DQN", dqn_agent)
    random_player1 = AIPlayer("Random1", random_agent1)
    random_player2 = AIPlayer("Random2", random_agent2)
    
    # Create the game
    game = PokerGame([dqn_player, random_player1, random_player2])
    
    # Training tracking
    episode_rewards = []
    episode_losses = []
    td_errors = []
    win_rates = []
    episodes = []
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    
    # DataFrame for detailed metrics
    metrics_df = pd.DataFrame(columns=[
        'episode', 'reward', 'loss', 'epsilon', 'win_rate', 
        'avg_chips', 'hands_played', 'fold_rate', 'bet_rate'
    ])
    
    # Early stopping tracking
    eval_rewards = deque(maxlen=5)  # Track last 5 evaluation rewards
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        # Reset game and players
        game.start_hand()
        dqn_player.chips = 1000
        random_player1.chips = 1000
        random_player2.chips = 1000
        
        episode_reward = 0
        episode_loss = 0
        episode_td_error = 0
        steps = 0
        hands_played = 0
        fold_count = 0
        bet_count = 0
        
        # Store initial chips for reward calculation
        initial_chips = dqn_player.chips
        
        # Play one complete hand
        while game.betting_round != 'showdown':
            current_player = game.current_player()
            
            if current_player == dqn_player:
                # DQN player's turn
                state = game.get_state(for_player=dqn_player)
                
                # Determine valid actions
                valid_actions = []
                if game.current_bet > dqn_player.current_bet:
                    valid_actions = ['fold', 'call']
                    if dqn_player.chips > game.current_bet - dqn_player.current_bet:
                        valid_actions.append('raise')
                else:
                    valid_actions = ['check']
                    if dqn_player.chips > 0:
                        valid_actions.append('bet')
                
                # Get action from DQN agent
                action_type, amount = dqn_agent.select_action(state, valid_actions, dqn_player)
                
                # Track action statistics
                if action_type == 'fold':
                    fold_count += 1
                elif action_type in ['bet', 'raise']:
                    bet_count += 1
                
                # Process the action
                try:
                    # Remember the state before action
                    prev_state = dqn_agent._preprocess_state(state, dqn_player)
                    prev_chips = dqn_player.chips
                    
                    # Execute the action
                    game.process_action(action_type, amount)
                    
                    # Get new state
                    next_state = game.get_state(for_player=dqn_player)
                    new_state_vec = dqn_agent._preprocess_state(next_state, dqn_player)
                    
                    # Calculate reward (change in chips)
                    if action_type == 'fold':
                        reward = -0.5  # Penalty for folding
                    else:
                        # Reward based on chip difference
                        reward = (dqn_player.chips - prev_chips) / 100.0
                    
                    # Check if betting round is over
                    done = game.is_betting_round_over()
                    
                    # Store experience
                    action_idx = ACTION_SPACE.index(action_type)
                    dqn_agent.remember(prev_state, action_idx, reward, new_state_vec, done)
                    
                    # Train the agent
                    loss = dqn_agent.train()
                    if loss is not None:
                        episode_loss += loss
                        # We would track TD errors here if they were accessible
                    
                    episode_reward += reward
                    steps += 1
                    
                except ValueError as e:
                    logger.error(f"Error processing action: {e}")
                    # Skip to next player
                    game.next_player()
            else:
                # Random agent's turn
                state = game.get_state(for_player=current_player)
                
                # Determine valid actions
                valid_actions = []
                if game.current_bet > current_player.current_bet:
                    valid_actions = ['fold', 'call']
                    if current_player.chips > game.current_bet - current_player.current_bet:
                        valid_actions.append('raise')
                else:
                    valid_actions = ['check']
                    if current_player.chips > 0:
                        valid_actions.append('bet')
                
                # Get action from the player's agent
                action_type, amount = current_player.act(state, valid_actions)
                
                # Process the action
                try:
                    game.process_action(action_type, amount)
                except ValueError as e:
                    logger.error(f"Error processing action: {e}")
                    # Skip to next player
                    game.next_player()
            
            # Check if betting round is over
            if game.is_betting_round_over():
                if game.betting_round == 'showdown':
                    # Hand is complete
                    hands_played += 1
                    winners = game.showdown()
                    
                    # Calculate final reward
                    final_reward = (dqn_player.chips - initial_chips) / 100.0
                    episode_reward += final_reward
                    
                    # Log results
                    if episode % 10 == 0:
                        logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, DQN chips = {dqn_player.chips}")
                        if dqn_player in winners:
                            logger.info("DQN player won!")
                        else:
                            logger.info("DQN player lost.")
                else:
                    # Move to next betting round
                    game.next_betting_round()
        
        # Record episode data
        episodes.append(episode)
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / max(1, steps))
        
        # Calculate additional metrics
        fold_rate = fold_count / max(1, steps) * 100
        bet_rate = bet_count / max(1, steps) * 100
        
        # Add to metrics DataFrame
        metrics_df = metrics_df.append({
            'episode': episode,
            'reward': episode_reward,
            'loss': episode_loss / max(1, steps),
            'epsilon': dqn_agent.epsilon,
            'win_rate': 100 if dqn_player in winners else 0,  # Simplified for this hand
            'avg_chips': dqn_player.chips,
            'hands_played': hands_played,
            'fold_rate': fold_rate,
            'bet_rate': bet_rate
        }, ignore_index=True)
        
        # Decay epsilon
        if dqn_agent.epsilon > epsilon_end:
            dqn_agent.epsilon *= epsilon_decay
            dqn_agent.epsilon = max(dqn_agent.epsilon, epsilon_end)
            if episode % 50 == 0:
                logger.info(f"Epsilon decayed to {dqn_agent.epsilon:.4f}")
        
        # Anneal beta parameter for prioritized experience replay
        if hasattr(dqn_agent.memory, 'beta'):
            beta_increment = (beta_end - beta_start) / num_episodes
            dqn_agent.memory.beta = min(beta_end, dqn_agent.memory.beta + beta_increment)
        
        # Save the model periodically
        if episode % save_freq == 0:
            model_path = os.path.join(model_dir, f"dqn_agent_episode_{episode}.pth")
            dqn_agent.save(model_path)
            logger.info(f"Model saved at episode {episode}")
            
            # Save metrics
            metrics_path = os.path.join(run_dir, f"metrics_episode_{episode}.csv")
            metrics_df.to_csv(metrics_path, index=False)
        
        # Evaluate the agent periodically
        if episode % eval_freq == 0:
            eval_reward, eval_metrics = evaluate_agent(dqn_agent, num_games=100)
            eval_rewards.append(eval_reward)
            win_rates.append(eval_metrics['win_rate'])
            
            logger.info(f"Evaluation at episode {episode}:")
            logger.info(f"  Average reward: {eval_reward:.2f}")
            logger.info(f"  Win rate: {eval_metrics['win_rate']:.2f}%")
            logger.info(f"  Average chips: {eval_metrics['avg_chips']:.2f}")
            logger.info(f"  Average hands: {eval_metrics['avg_hands']:.2f}")
            
            # Check for early stopping
            if early_stopping:
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    no_improvement_count = 0
                    # Save best model
                    dqn_agent.save(os.path.join(model_dir, "dqn_agent_best.pth"))
                    logger.info(f"New best model saved with reward {best_eval_reward:.2f}")
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} evaluations")
                    
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping triggered after {episode} episodes")
                    break
        
        # Plot progress periodically
        if episode % 100 == 0 and episode > 0:
            plot_path = os.path.join(run_dir, f"training_plot_episode_{episode}.png")
            plot_detailed_training_progress(
                episodes, episode_rewards, episode_losses, win_rates, 
                metrics_df, save_path=plot_path
            )
    
    # Save the final model
    dqn_agent.save(os.path.join(model_dir, "dqn_agent_final.pth"))
    
    # Plot final training progress
    final_plot_path = os.path.join(run_dir, "training_plot_final.png")
    plot_detailed_training_progress(
        episodes, episode_rewards, episode_losses, win_rates, 
        metrics_df, save_path=final_plot_path
    )
    
    # Save final metrics
    final_metrics_path = os.path.join(run_dir, "metrics_final.csv")
    metrics_df.to_csv(final_metrics_path, index=False)
    
    return dqn_agent

def evaluate_agent(agent, num_games=100):
    """
    Evaluate an agent's performance with detailed metrics.
    
    Args:
        agent (BaseAgent): The agent to evaluate
        num_games (int, optional): Number of games to play. Defaults to 100.
        
    Returns:
        tuple: (average_reward, metrics_dict) - Average reward and dictionary of metrics
    """
    # Create a temporary copy of the agent with no exploration
    eval_agent = DQNAgent(
        input_dim=agent.input_dim,
        action_space=agent.action_space,
        epsilon=0.0,  # No exploration during evaluation
        n_steps=agent.n_steps,
        alpha=agent.memory.alpha if hasattr(agent.memory, 'alpha') else 0.6,
        beta=1.0  # Full importance sampling correction
    )
    eval_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
    
    # Create random opponents
    random_agent1 = RandomAgent()
    random_agent2 = RandomAgent()
    
    # Create players
    eval_player = AIPlayer("Eval", eval_agent)
    random_player1 = AIPlayer("Random1", random_agent1)
    random_player2 = AIPlayer("Random2", random_agent2)
    
    # Create the game
    game = PokerGame([eval_player, random_player1, random_player2])
    
    # Metrics tracking
    total_reward = 0
    wins = 0
    total_chips = 0
    total_hands = 0
    fold_count = 0
    bet_count = 0
    call_count = 0
    check_count = 0
    
    for game_num in range(num_games):
        # Reset game and players
        game.start_hand()
        eval_player.chips = 1000
        random_player1.chips = 1000
        random_player2.chips = 1000
        
        initial_chips = eval_player.chips
        hands_in_game = 0
        
        # Play one complete hand
        while game.betting_round != 'showdown':
            current_player = game.current_player()
            
            state = game.get_state(for_player=current_player)
            
            # Determine valid actions
            valid_actions = []
            if game.current_bet > current_player.current_bet:
                valid_actions = ['fold', 'call']
                if current_player.chips > game.current_bet - current_player.current_bet:
                    valid_actions.append('raise')
            else:
                valid_actions = ['check']
                if current_player.chips > 0:
                    valid_actions.append('bet')
            
            # Get action from the player's agent
            action_type, amount = current_player.act(state, valid_actions)
            
            # Track agent actions
            if current_player == eval_player:
                if action_type == 'fold':
                    fold_count += 1
                elif action_type == 'check':
                    check_count += 1
                elif action_type == 'call':
                    call_count += 1
                elif action_type in ['bet', 'raise']:
                    bet_count += 1
            
            # Process the action
            try:
                game.process_action(action_type, amount)
            except ValueError:
                # Skip to next player on error
                game.next_player()
            
            # Check if betting round is over
            if game.is_betting_round_over():
                if game.betting_round == 'showdown':
                    # Hand is complete
                    hands_in_game += 1
                    total_hands += 1
                    winners = game.showdown()
                    
                    # Calculate reward
                    reward = eval_player.chips - initial_chips
                    total_reward += reward
                    total_chips += eval_player.chips
                    
                    if eval_player in winners:
                        wins += 1
                else:
                    # Move to next betting round
                    game.next_betting_round()
    
    # Calculate metrics
    avg_reward = total_reward / num_games
    win_rate = (wins / total_hands) * 100 if total_hands > 0 else 0
    avg_chips = total_chips / num_games
    avg_hands = total_hands / num_games
    
    # Action distribution
    total_actions = fold_count + check_count + call_count + bet_count
    fold_pct = (fold_count / total_actions) * 100 if total_actions > 0 else 0
    check_pct = (check_count / total_actions) * 100 if total_actions > 0 else 0
    call_pct = (call_count / total_actions) * 100 if total_actions > 0 else 0
    bet_pct = (bet_count / total_actions) * 100 if total_actions > 0 else 0
    
    logger.info(f"Evaluation: Wins = {wins}/{total_hands} ({win_rate:.1f}%)")
    logger.info(f"Action distribution: Fold={fold_pct:.1f}%, Check={check_pct:.1f}%, Call={call_pct:.1f}%, Bet/Raise={bet_pct:.1f}%")
    
    metrics = {
        'win_rate': win_rate,
        'avg_chips': avg_chips,
        'avg_hands': avg_hands,
        'fold_pct': fold_pct,
        'check_pct': check_pct,
        'call_pct': call_pct,
        'bet_pct': bet_pct
    }
    
    return avg_reward, metrics

def plot_detailed_training_progress(episodes, rewards, losses, win_rates, metrics_df, save_path=None):
    """
    Plot detailed training progress with multiple metrics.
    
    Args:
        episodes (list): Episode numbers
        rewards (list): Episode rewards
        losses (list): Episode losses
        win_rates (list): Evaluation win rates
        metrics_df (pd.DataFrame): DataFrame with detailed metrics
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Plot rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    
    # Plot losses
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, losses)
    ax2.set_title("Training Losses")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    
    # Plot win rate over time (from evaluations)
    ax3 = fig.add_subplot(gs[1, 0])
    eval_episodes = [episodes[i] for i in range(0, len(episodes), len(episodes)//len(win_rates))][:len(win_rates)]
    ax3.plot(eval_episodes, win_rates)
    ax3.set_title("Win Rate (Evaluation)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Win Rate (%)")
    
    # Plot epsilon decay
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics_df['episode'], metrics_df['epsilon'])
    ax4.set_title("Exploration Rate (Epsilon)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    
    # Plot action distribution
    ax5 = fig.add_subplot(gs[2, 0])
    if 'fold_rate' in metrics_df.columns and 'bet_rate' in metrics_df.columns:
        # Smooth the data for better visualization
        window_size = min(50, len(metrics_df) // 10)
        fold_rate_smooth = metrics_df['fold_rate'].rolling(window=window_size, min_periods=1).mean()
        bet_rate_smooth = metrics_df['bet_rate'].rolling(window=window_size, min_periods=1).mean()
        
        ax5.plot(metrics_df['episode'], fold_rate_smooth, label='Fold %')
        ax5.plot(metrics_df['episode'], bet_rate_smooth, label='Bet/Raise %')
        ax5.set_title("Action Distribution Over Time")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Percentage")
        ax5.legend()
    
    # Plot average chips
    ax6 = fig.add_subplot(gs[2, 1])
    if 'avg_chips' in metrics_df.columns:
        # Smooth the data
        avg_chips_smooth = metrics_df['avg_chips'].rolling(window=window_size, min_periods=1).mean()
        ax6.plot(metrics_df['episode'], avg_chips_smooth)
        ax6.set_title("Average Chips")
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Chips")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train a poker RL agent')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='How often to save the model')
    parser.add_argument('--eval-freq', type=int, default=50, help='How often to evaluate the model')
    
    # RL algorithm parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=10000, help='Size of replay buffer')
    
    # Exploration parameters
    parser.add_argument('--epsilon-start', type=float, default=0.3, help='Starting exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Decay rate for exploration')
    
    # Advanced RL parameters
    parser.add_argument('--target-update', type=int, default=10, help='How often to update target network')
    parser.add_argument('--alpha', type=float, default=0.6, help='Prioritization exponent for PER')
    parser.add_argument('--beta-start', type=float, default=0.4, help='Initial importance sampling correction')
    parser.add_argument('--beta-end', type=float, default=1.0, help='Final importance sampling correction')
    parser.add_argument('--n-steps', type=int, default=3, help='Steps for n-step returns')
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    # Output directories
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--model-dir', type=str, default='data', help='Directory for saved models')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parse command line arguments
    args = parse_args()
    
    # Log the configuration
    logger.info("Starting training with configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Train the agent
    trained_agent = train_agent(
        num_episodes=args.episodes,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        alpha=args.alpha,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        n_steps=args.n_steps,
        early_stopping=args.early_stopping,
        patience=args.patience,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )
    
    # Final evaluation
    final_reward, final_metrics = evaluate_agent(trained_agent, num_games=200)
    logger.info(f"Final evaluation:")
    logger.info(f"  Average reward: {final_reward:.2f}")
    logger.info(f"  Win rate: {final_metrics['win_rate']:.2f}%")
    logger.info(f"  Average chips: {final_metrics['avg_chips']:.2f}")
    logger.info(f"  Action distribution: Fold={final_metrics['fold_pct']:.1f}%, "
               f"Check={final_metrics['check_pct']:.1f}%, "
               f"Call={final_metrics['call_pct']:.1f}%, "
               f"Bet/Raise={final_metrics['bet_pct']:.1f}%")
