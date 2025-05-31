import streamlit as st
import time
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from poker.game import PokerGame
from poker.player import HumanPlayer, AIPlayer
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from utils import evaluate_hand, calculate_pot_odds, plot_training_progress

# Action space definition (must match the one in training.py)
ACTION_SPACE = ['fold', 'check', 'call', 'bet', 'raise']

def initialize_game():
    """Initialize the poker game with players."""
    # Get agent type from session state
    agent_type = st.session_state.get('agent_type', 'random')
    
    # Create human player
    human = HumanPlayer("Human", chips=1000)
    
    # Create AI players
    ai_agents = []
    
    if agent_type == 'dqn':
        # Try to load trained DQN agent
        model_path = st.session_state.get('model_path', 'data/dqn_agent_final.pth')
        if os.path.exists(model_path):
            state_dim = 50  # Must match the dimension used in training
            
            # Get advanced parameters from session state (or use defaults)
            n_steps = st.session_state.get('n_steps', 3)
            alpha = st.session_state.get('alpha', 0.6)
            beta = st.session_state.get('beta', 1.0)  # Use 1.0 for evaluation
            
            # Create enhanced DQN agent with advanced features
            dqn_agent = DQNAgent(
                input_dim=state_dim, 
                action_space=ACTION_SPACE, 
                epsilon=0.0,  # No exploration during gameplay
                n_steps=n_steps,
                alpha=alpha,
                beta=beta
            )
            
            try:
                dqn_agent.load(model_path)
                ai_agents.append(AIPlayer("DQN Agent", dqn_agent, chips=1000))
                st.session_state['agent_loaded'] = True
                st.session_state['dqn_agent'] = dqn_agent  # Store for visualization
            except Exception as e:
                st.error(f"Failed to load DQN agent: {e}")
                agent_type = 'random'
                st.session_state['agent_type'] = 'random'
                st.session_state['agent_loaded'] = False
        else:
            st.warning(f"Model file not found: {model_path}")
            agent_type = 'random'
            st.session_state['agent_type'] = 'random'
            st.session_state['agent_loaded'] = False
    
    # Add random agents (either as the main agent or as opponents)
    if agent_type == 'random' or len(ai_agents) == 0:
        ai_agents.append(AIPlayer("Random Agent", RandomAgent(), chips=1000))
    
    # Add additional random opponent
    ai_agents.append(AIPlayer("Random Opponent", RandomAgent(), chips=1000))
    
    # Create and return the game
    return PokerGame([human] + ai_agents)

def display_card(card, key=None):
    """Display a single card with color based on suit."""
    if not card:
        return st.markdown("🂠", unsafe_allow_html=True)
    
    # Define colors for suits
    suit_colors = {
        'hearts': 'red',
        'diamonds': 'red',
        'clubs': 'black',
        'spades': 'black'
    }
    
    # Define suit symbols
    suit_symbols = {
        'hearts': '♥',
        'diamonds': '♦',
        'clubs': '♣',
        'spades': '♠'
    }
    
    if not card.visible:
        return st.markdown("<div style='text-align: center; font-size: 48px;'>🂠</div>", unsafe_allow_html=True)
    
    # Card styling
    color = suit_colors.get(card.suit, 'black')
    symbol = suit_symbols.get(card.suit, '?')
    
    # Display card
    st.markdown(
        f"<div style='text-align: center; border: 2px solid black; border-radius: 10px; padding: 10px; "
        f"width: 70px; height: 100px; margin: auto; background-color: white;'>"
        f"<div style='color: {color}; font-size: 24px; text-align: left;'>{card.RANKS[card.rank]}</div>"
        f"<div style='color: {color}; font-size: 36px; text-align: center;'>{symbol}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def display_cards(cards):
    """Display multiple cards in a row."""
    if not cards:
        return
    
    # Create columns for cards
    cols = st.columns(len(cards))
    for i, card in enumerate(cards):
        with cols[i]:
            display_card(card, key=f"card_{i}")

def display_game_state(game, human_player):
    """Display the current game state in the UI."""
    # Game info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Pot:** ${game.pot}")
    with col2:
        st.markdown(f"**Round:** {game.betting_round.capitalize()}")
    with col3:
        st.markdown(f"**Current Bet:** ${game.current_bet}")
    
    # Community cards
    st.markdown("### Community Cards")
    display_cards(game.community_cards)
    
    # Player info
    st.markdown("### Your Hand")
    display_cards(human_player.hand)
    st.markdown(f"**Chips:** ${human_player.chips}")
    
    # Hand strength
    if len(human_player.hand) == 2:
        hand_rank, _ = evaluate_hand(human_player.hand, game.community_cards)
        hand_names = ["High Card", "Pair", "Two Pair", "Three of a Kind", 
                     "Straight", "Flush", "Full House", "Four of a Kind", 
                     "Straight Flush", "Royal Flush"]
        if len(game.community_cards) > 0:  # Only show if community cards exist
            st.markdown(f"**Current Hand:** {hand_names[hand_rank]}")
    
    # Pot odds
    if game.current_bet > human_player.current_bet and human_player.chips > 0:
        call_amount = min(game.current_bet - human_player.current_bet, human_player.chips)
        odds = calculate_pot_odds(game.pot, call_amount)
        st.markdown(f"**Pot Odds:** {odds:.1f}%")
    
    # Other players
    st.markdown("### Players")
    for player in game.players:
        status = "Active" if player.is_active else "Folded"
        is_current = player == game.current_player()
        marker = "➡️ " if is_current else ""
        
        if player == human_player:
            continue  # Skip human player
        
        st.markdown(f"{marker}**{player.name}:** ${player.chips} - {status} - Current Bet: ${player.current_bet}")

def display_agent_decision_making(game_state, agent, player):
    """
    Display the agent's decision-making process with Q-values visualization.
    
    Args:
        game_state (dict): Current game state
        agent (DQNAgent): The DQN agent
        player (AIPlayer): The player using the agent
    """
    if not hasattr(agent, 'policy_net') or not hasattr(agent, 'action_space'):
        return
    
    st.markdown("### Agent Decision Analysis")
    
    # Process state
    state = agent._preprocess_state(game_state, player)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    # Get Q-values
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).squeeze().cpu().numpy()
    
    # If using dueling architecture, get value and advantage streams
    if hasattr(agent.policy_net, 'value_stream') and hasattr(agent.policy_net, 'advantage_stream'):
        features = agent.policy_net.feature_layer(state_tensor)
        value = agent.policy_net.value_stream(features).item()
        advantages = agent.policy_net.advantage_stream(features).squeeze().cpu().numpy()
        
        # Display value estimate
        st.markdown(f"**State Value Estimate:** {value:.2f}")
        
        # Create DataFrame for visualization
        data = {
            'Action': agent.action_space,
            'Q-Value': q_values,
            'Advantage': advantages
        }
        df = pd.DataFrame(data)
        
        # Determine valid actions
        valid_actions = []
        if game_state['current_bet'] > player.current_bet:
            valid_actions = ['fold', 'call']
            if player.chips > game_state['current_bet'] - player.current_bet:
                valid_actions.append('raise')
        else:
            valid_actions = ['check']
            if player.chips > 0:
                valid_actions.append('bet')
        
        # Mark valid actions
        df['Valid'] = df['Action'].apply(lambda x: 'Yes' if x in valid_actions else 'No')
        
        # Highlight best action
        best_valid_idx = df[df['Valid'] == 'Yes']['Q-Value'].idxmax()
        df['Best Action'] = [i == best_valid_idx for i in range(len(df))]
        
        # Display as table
        st.dataframe(df.style.apply(lambda x: ['background-color: lightgreen' if x['Best Action'] else '' for i in x], axis=1))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df['Action'], df['Q-Value'], color=['green' if x else 'gray' for x in df['Best Action']])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax.set_title('Q-Values by Action')
        ax.set_ylabel('Q-Value')
        st.pyplot(fig)
        
        # Create advantage visualization
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(df['Action'], df['Advantage'])
        ax2.set_title('Action Advantages')
        ax2.set_ylabel('Advantage')
        st.pyplot(fig2)
    else:
        # For non-dueling networks, just show Q-values
        data = {
            'Action': agent.action_space,
            'Q-Value': q_values
        }
        df = pd.DataFrame(data)
        st.dataframe(df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df['Action'], df['Q-Value'])
        ax.set_title('Q-Values by Action')
        ax.set_ylabel('Q-Value')
        st.pyplot(fig)

def get_human_action(game, human_player):
    """Get action from human player through UI."""
    valid_actions = []
    
    # Determine valid actions
    if game.current_bet > human_player.current_bet:
        valid_actions = ['fold', 'call']
        if human_player.chips > game.current_bet - human_player.current_bet:
            valid_actions.append('raise')
    else:
        valid_actions = ['check']
        if human_player.chips > 0:
            valid_actions.append('bet')
    
    # Action selection
    action = st.radio("Choose your action:", valid_actions)
    
    # Amount if needed
    amount = 0
    if action in ['bet', 'raise']:
        min_amount = max(game.current_bet * 2, game.big_blind) if action == 'raise' else game.big_blind
        max_amount = human_player.chips
        amount = st.slider("Amount:", min_value=min_amount, max_value=max_amount, value=min_amount)
    
    # Confirm action
    if st.button("Confirm Action"):
        return action, amount
    
    return None, None

def display_winners(winners):
    """Display the winners of the hand."""
    st.markdown("### Hand Complete!")
    
    if len(winners) == 1:
        winner = winners[0]
        st.success(f"**{winner.name}** wins the pot!")
    else:
        st.success(f"Split pot between: {', '.join([w.name for w in winners])}")
    
    # Display final hands
    for player in st.session_state.game.players:
        st.markdown(f"**{player.name}'s Hand:**")
        display_cards(player.hand)

def agent_settings():
    """Settings page for agent configuration."""
    st.title("Agent Settings")
    
    # Agent type selection
    agent_type = st.radio(
        "Select AI Agent Type:",
        ["random", "dqn"],
        index=0 if st.session_state.get('agent_type') == 'random' else 1
    )
    
    # Model selection for DQN
    if agent_type == 'dqn':
        # Check if data directory exists
        if not os.path.exists('data'):
            st.warning("No 'data' directory found. Please train a DQN agent first.")
        else:
            # List available model files
            model_files = [f for f in os.listdir('data') if f.endswith('.pth')]
            if model_files:
                model_path = st.selectbox(
                    "Select Model:",
                    model_files,
                    index=0
                )
                st.session_state['model_path'] = os.path.join('data', model_path)
                
                # Advanced RL configuration
                st.markdown("### Advanced RL Settings")
                
                # N-step returns
                n_steps = st.slider(
                    "N-step Returns:", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state.get('n_steps', 3),
                    help="Number of steps for n-step returns. Higher values look further into the future."
                )
                st.session_state['n_steps'] = n_steps
                
                # Prioritization exponent
                alpha = st.slider(
                    "Prioritization Exponent (Alpha):", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get('alpha', 0.6),
                    step=0.1,
                    help="Controls how much prioritization is used (0 = uniform, 1 = full prioritization)"
                )
                st.session_state['alpha'] = alpha
                
                # Beta parameter
                beta = st.slider(
                    "Importance Sampling (Beta):", 
                    min_value=0.4, 
                    max_value=1.0, 
                    value=st.session_state.get('beta', 1.0),
                    step=0.1,
                    help="Importance sampling correction factor (0.4 = minimal correction, 1.0 = full correction)"
                )
                st.session_state['beta'] = beta
                
                # Display architecture info
                st.markdown("### Model Architecture")
                st.info("""
                This agent uses a Dueling DQN architecture which separates state value and action advantage estimation.
                It also implements Double DQN to reduce overestimation bias and Prioritized Experience Replay for more
                efficient learning from important experiences.
                """)
            else:
                st.warning("No model files found in 'data' directory. Please train a DQN agent first.")
    
    # Save settings
    if st.button("Save Settings"):
        st.session_state['agent_type'] = agent_type
        st.success("Settings saved!")
        # Force game reinitialization
        if 'game' in st.session_state:
            del st.session_state['game']
        st.rerun()

def training_page():
    """Page for training the RL agent."""
    st.title("Train RL Agent")
    
    # Basic training parameters
    st.markdown("### Basic Parameters")
    num_episodes = st.slider("Number of Episodes", min_value=100, max_value=5000, value=1000, step=100)
    save_freq = st.slider("Save Frequency (episodes)", min_value=10, max_value=500, value=100, step=10)
    eval_freq = st.slider("Evaluation Frequency (episodes)", min_value=10, max_value=500, value=50, step=10)
    
    # Advanced RL parameters
    st.markdown("### Advanced RL Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Learning parameters
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
        gamma = st.slider("Discount Factor (Gamma)", min_value=0.8, max_value=0.999, value=0.99, step=0.001, format="%.3f")
        batch_size = st.slider("Batch Size", min_value=16, max_value=256, value=32, step=16)
        memory_size = st.slider("Memory Size", min_value=1000, max_value=100000, value=10000, step=1000)
        
    with col2:
        # Exploration parameters
        epsilon_start = st.slider("Starting Exploration (Epsilon)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        epsilon_end = st.slider("Final Exploration", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        epsilon_decay = st.slider("Epsilon Decay Rate", min_value=0.9, max_value=0.999, value=0.995, step=0.001, format="%.3f")
    
    # Dueling DQN parameters
    st.markdown("### Dueling DQN Parameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Target network update frequency
        target_update = st.slider("Target Network Update Frequency", min_value=1, max_value=100, value=10, step=1)
        
        # N-step returns
        n_steps = st.slider("N-step Returns", min_value=1, max_value=10, value=3, step=1)
    
    with col4:
        # Prioritized Experience Replay parameters
        alpha = st.slider("Prioritization Exponent (Alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        beta_start = st.slider("Initial Importance Sampling (Beta)", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
        beta_end = st.slider("Final Importance Sampling", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    
    # Early stopping
    st.markdown("### Training Control")
    early_stopping = st.checkbox("Enable Early Stopping", value=True)
    patience = st.slider("Early Stopping Patience", min_value=5, max_value=200, value=100, step=5, 
                        help="Number of evaluations without improvement before stopping")
    
    # Output directories
    st.markdown("### Output Settings")
    log_dir = st.text_input("Log Directory", value="logs")
    model_dir = st.text_input("Model Directory", value="data")
    
    # Store all parameters in session state
    training_params = {
        'num_episodes': num_episodes,
        'save_freq': save_freq,
        'eval_freq': eval_freq,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'batch_size': batch_size,
        'memory_size': memory_size,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'target_update': target_update,
        'n_steps': n_steps,
        'alpha': alpha,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'early_stopping': early_stopping,
        'patience': patience,
        'log_dir': log_dir,
        'model_dir': model_dir
    }
    
    # Start training
    if st.button("Start Training"):
        st.session_state['training'] = True
        st.session_state['train_params'] = training_params
        st.rerun()
    
    # Training progress
    if st.session_state.get('training', False):
        params = st.session_state.get('train_params', {})
        
        # Display progress
        st.markdown("### Training in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Import here to avoid circular imports
        from training import train_agent
        
        # Placeholder for plots
        plot_container = st.empty()
        
        # Mock training progress (in a real app, you'd run training in a separate thread/process)
        st.warning("This is a simplified training visualization. In a real application, you would run training in a background process.")
        
        episode_rewards = []
        episode_losses = []
        episodes = []
        
        for i in range(min(100, params.get('num_episodes', 1000))):
            # Update progress
            progress = (i + 1) / min(100, params.get('num_episodes', 1000))
            progress_bar.progress(progress)
            status_text.text(f"Episode {i+1}/{min(100, params.get('num_episodes', 1000))}")
            
            # Simulate training results
            reward = np.random.normal(-5, 20)  # Random reward with some noise
            loss = max(0, 10 - i/10) + np.random.normal(0, 1)  # Decreasing loss with noise
            
            episodes.append(i)
            episode_rewards.append(reward)
            episode_losses.append(loss)
            
            # Plot occasionally
            if (i + 1) % 10 == 0:
                with plot_container.container():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(episodes, episode_rewards)
                    ax1.set_title("Rewards")
                    ax1.set_xlabel("Episode")
                    ax1.set_ylabel("Reward")
                    
                    ax2.plot(episodes, episode_losses)
                    ax2.set_title("Losses")
                    ax2.set_xlabel("Episode")
                    ax2.set_ylabel("Loss")
                    
                    st.pyplot(fig)
            
            # Simulate time passing
            time.sleep(0.05)
        
        # Finish training
        progress_bar.progress(1.0)
        status_text.text("Training complete!")
        st.session_state['training'] = False
        st.success("Training complete! The model has been saved to the 'data' directory.")

def main():
    """Main application entry point."""
    st.sidebar.title("Poker RL Agent")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Play", "Train Agent", "Settings"])
    
    if page == "Play":
        st.title("Poker RL Agent")
        
        # Initialize game if not exists
        if 'game' not in st.session_state:
            st.session_state.game = initialize_game()
            st.session_state.game.start_hand()
        
        game = st.session_state.game
        human_player = game.players[0]  # First player is human
        
        # Display game state
        display_game_state(game, human_player)
        
        # Display agent decision making if DQN agent is loaded
        if st.session_state.get('agent_loaded', False) and st.session_state.get('dqn_agent') is not None:
            # Find the DQN player
            dqn_player = None
            for player in game.players:
                if "DQN" in player.name:
                    dqn_player = player
                    break
            
            if dqn_player:
                # Get current game state
                state = game.get_state(for_player=dqn_player)
                display_agent_decision_making(state, st.session_state.get('dqn_agent'), dqn_player)
        
        # Handle player actions
        if game.current_player() == human_player:
            st.markdown("### Your Turn")
            action_type, amount = get_human_action(game, human_player)
            
            if action_type:
                try:
                    # Process the player's action
                    game.process_action(action_type, amount)
                    
                    # Handle end of betting round
                    if game.is_betting_round_over():
                        # If we're at showdown, determine winner
                        if game.betting_round == 'showdown':
                            winners = game.showdown()
                            display_winners(winners)
                            if st.button("Start New Hand", key="new_hand_human"):
                                game.start_hand()
                                st.rerun()
                            # Return early to prevent further processing
                            return
                            
                        # If we're at river, transition to showdown
                        elif game.betting_round == 'river':
                            winners = game.next_betting_round()  # This should go to showdown
                            display_winners(winners)
                            if st.button("Start New Hand", key="new_hand_river"):
                                game.start_hand()
                                st.rerun()
                            # Return early to prevent further processing
                            return
                            
                        # Otherwise move to next betting round
                        else:
                            game.next_betting_round()
                    
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            st.markdown(f"### {game.current_player().name}'s Turn")
            
            # In a real implementation, we'd have a proper game loop
            # For simplicity, we'll just simulate AI action on button press
            if st.button("Continue"):
                ai_player = game.current_player()
                state = game.get_state(for_player=ai_player)
                
                # Determine valid actions
                valid_actions = []
                if game.current_bet > ai_player.current_bet:
                    valid_actions = ['fold', 'call']
                    if ai_player.chips > game.current_bet - ai_player.current_bet:
                        valid_actions.append('raise')
                else:
                    valid_actions = ['check']
                    if ai_player.chips > 0:
                        valid_actions.append('bet')
                
                action_type, amount = ai_player.act(state, valid_actions)
                
                # Show what the AI did
                st.info(f"{ai_player.name} {action_type}s" + (f" ${amount}" if amount > 0 else ""))
                
                # Process the AI's action
                game.process_action(action_type, amount)
                
                # Handle end of betting round
                if game.is_betting_round_over():
                    # If we're at showdown, determine winner
                    if game.betting_round == 'showdown':
                        winners = game.showdown()
                        display_winners(winners)
                        if st.button("Start New Hand", key="new_hand_ai"):
                            game.start_hand()
                            st.rerun()
                        # Return early to prevent further processing
                        return
                        
                    # If we're at river, transition to showdown
                    elif game.betting_round == 'river':
                        winners = game.next_betting_round()  # This should go to showdown
                        display_winners(winners)
                        if st.button("Start New Hand", key="new_hand_ai_river"):
                            game.start_hand()
                            st.rerun()
                        # Return early to prevent further processing
                        return
                        
                    # Otherwise move to next betting round
                    else:
                        game.next_betting_round()
                
                st.rerun()
    
    elif page == "Train Agent":
        training_page()
    
    elif page == "Settings":
        agent_settings()

if __name__ == "__main__":
    # Initialize session state
    if 'agent_type' not in st.session_state:
        st.session_state['agent_type'] = 'random'
    
    if 'agent_loaded' not in st.session_state:
        st.session_state['agent_loaded'] = False
    
    main()
