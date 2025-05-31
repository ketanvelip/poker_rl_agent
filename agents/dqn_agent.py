import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from .base_agent import BaseAgent

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for poker decision making.
    
    Separates state value and action advantages for better policy evaluation.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Initialize the Dueling DQN.
        
        Args:
            input_dim (int): Dimension of the state representation
            output_dim (int): Number of possible actions
            hidden_dim (int, optional): Size of hidden layers. Defaults to 128.
        """
        super().__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream - estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream - estimates advantages for each action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the dueling network architecture.
        
        Args:
            x (torch.Tensor): Input tensor representing the state
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        features = self.feature_layer(x)
        
        # Calculate state value
        value = self.value_stream(features)
        
        # Calculate action advantages
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This helps with stability by forcing the advantage mean to be zero
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for storing and sampling experiences.
    
    Uses TD errors as priorities for more efficient learning.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
            alpha (float): Determines how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta (float): Importance-sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment (float): Increment for beta parameter annealing
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.epsilon = 1e-5  # Small constant to ensure non-zero priorities
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of experiences, importance sampling weights, and indices
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Anneal beta toward 1 (full importance sampling correction)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (states, actions, rewards, next_states, dones), weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices (list): Indices of experiences to update
            priorities (list): New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon  # Add epsilon to ensure non-zero
    
    def __len__(self):
        """Get the current size of the buffer."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Enhanced DQN Agent with advanced reinforcement learning techniques.
    
    Features:
    - Dueling DQN architecture
    - Double DQN for reduced overestimation
    - Prioritized Experience Replay
    - N-step returns for more efficient learning
    
    Attributes:
        epsilon (float): Exploration rate
        gamma (float): Discount factor
        input_dim (int): Dimension of the state representation
        action_space (list): List of possible actions
        device (torch.device): Device to run the network on
        policy_net (DuelingDQN): Policy network
        target_net (DuelingDQN): Target network
        optimizer (torch.optim): Optimizer for the policy network
        memory (PrioritizedReplayBuffer): Prioritized replay buffer
        n_step_buffer (deque): Buffer for n-step returns
    """
    
    def __init__(self, input_dim, action_space, epsilon=0.1, gamma=0.99, 
                 learning_rate=0.001, memory_size=10000, batch_size=32, 
                 target_update=10, alpha=0.6, beta=0.4, n_steps=3):
        """
        Initialize the enhanced DQN agent.
        
        Args:
            input_dim (int): Dimension of the state representation
            action_space (list): List of possible actions
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            memory_size (int, optional): Size of replay memory. Defaults to 10000.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            target_update (int, optional): Frequency of target network update. Defaults to 10.
            alpha (float, optional): Prioritization exponent. Defaults to 0.6.
            beta (float, optional): Importance sampling correction factor. Defaults to 0.4.
            n_steps (int, optional): Number of steps for n-step returns. Defaults to 3.
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_dim = input_dim
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        self.n_steps = n_steps
        
        # Networks - using dueling architecture
        self.policy_net = DuelingDQN(input_dim, len(action_space)).to(self.device)
        self.target_net = DuelingDQN(input_dim, len(action_space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_size,
            alpha=alpha,
            beta=beta
        )
        
        # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def _get_n_step_info(self, n_step_buffer, gamma):
        """
        Calculate the n-step returns.
        
        Args:
            n_step_buffer (deque): Buffer containing n transitions
            gamma (float): Discount factor
            
        Returns:
            tuple: First state, action, n-step return, final state, and done flag
        """
        state, action = n_step_buffer[0][:2]
        
        # Calculate n-step return
        reward = 0
        for i, (_, _, r, _, _) in enumerate(n_step_buffer):
            reward += (gamma ** i) * r
        
        # Get the final state and done flag
        _, _, _, next_state, done = n_step_buffer[-1]
        
        return state, action, reward, next_state, done
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in n-step buffer and memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n-step buffer is ready, process it
        if len(self.n_step_buffer) >= self.n_steps:
            state, action, reward, next_state, done = self._get_n_step_info(
                self.n_step_buffer, self.gamma
            )
            self.memory.add(state, action, reward, next_state, done)
    
    def select_action(self, game_state, valid_actions, player):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            game_state (dict): The current state of the game
            valid_actions (list): List of valid actions
            player (Player): The player making the decision
            
        Returns:
            tuple: (action_type, amount)
        """
        # Convert game state to tensor
        state = self._preprocess_state(game_state, player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(len(self.action_space))
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
        
        action_type = self.action_space[action_idx]
        
        # Handle bet/raise amount
        amount = 0
        if action_type in ['bet', 'raise']:
            min_bet = game_state['current_bet'] * 2
            max_bet = player.chips
            amount = min_bet  # For simplicity, just use min bet
        
        return (action_type, amount)
    
    def train(self):
        """
        Train the agent using prioritized replay and Double DQN.
        
        Returns:
            float: Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch with priorities
        batch, weights, indices = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_tensor).gather(1, action_tensor)
        
        # Compute next Q values using Double DQN
        with torch.no_grad():
            # Select actions using policy network
            next_action_indices = self.policy_net(next_state_tensor).max(1)[1].unsqueeze(1)
            
            # Evaluate Q values using target network
            next_q_values = self.target_net(next_state_tensor).gather(1, next_action_indices).squeeze(1)
            
            # Compute target Q values
            target_q_values = reward_tensor + (1 - done_tensor) * (self.gamma ** self.n_steps) * next_q_values
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Compute Huber loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights_tensor).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return weighted_loss.item()
    
    def _preprocess_state(self, game_state, player):
        """
        Convert game state to a vector representation.
        
        Args:
            game_state (dict): The current state of the game
            player (Player): The player making the decision
            
        Returns:
            list: Vector representation of the state
        """
        # Simplified state representation
        # This is a placeholder and should be expanded for better performance
        state = []
        
        # Pot and current bet
        state.append(game_state['pot'] / 1000.0)  # Normalize
        state.append(game_state['current_bet'] / 100.0)  # Normalize
        
        # Player info
        state.append(player.chips / 1000.0)  # Normalize
        
        # Card encodings (simplified)
        for card in player.hand:
            state.append(card.rank / 14.0)  # Normalize rank
            for suit_idx, suit in enumerate(card.SUITS):
                state.append(1.0 if card.suit == suit else 0.0)
        
        # Community cards
        for card in game_state['community_cards']:
            state.append(card.rank / 14.0)  # Normalize rank
            for suit_idx, suit in enumerate(card.SUITS):
                state.append(1.0 if card.suit == suit else 0.0)
        
        # Pad for missing community cards
        missing_cards = 5 - len(game_state['community_cards'])
        for _ in range(missing_cards):
            state.extend([0.0] * 5)  # 1 for rank + 4 for suits
        
        # Game state
        betting_rounds = ['preflop', 'flop', 'turn', 'river', 'showdown']
        for br in betting_rounds:
            state.append(1.0 if game_state['betting_round'] == br else 0.0)
        
        return state
    
    def save(self, filename):
        """
        Save the agent's policy network.
        
        Args:
            filename (str): Filename to save to
        """
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        """
        Load the agent's policy network.
        
        Args:
            filename (str): Filename to load from
        """
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())
