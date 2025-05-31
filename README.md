# Poker RL Agent

A reinforcement learning agent that learns to play poker through self-play and competitive training, featuring a Streamlit user interface for interactive gameplay.

## Features

- Texas Hold'em poker game environment
- Deep Q-Network (DQN) reinforcement learning agent
- Random baseline agent for comparison
- Interactive Streamlit UI to play against trained agents
- Training visualization and analytics
- Hand evaluation and poker strategy utilities

## Project Structure

```
poker_rl_agent/
├── poker/               # Poker game implementation
│   ├── card.py          # Card and deck classes
│   ├── game.py          # Texas Hold'em game logic
│   └── player.py        # Player implementations
├── agents/              # RL agents
│   ├── base_agent.py    # Abstract agent interface
│   ├── dqn_agent.py     # Deep Q-Network implementation
│   └── random_agent.py  # Random baseline agent
├── data/                # Trained models and logs
├── app.py               # Streamlit UI
├── training.py          # Agent training pipeline
├── utils.py             # Utility functions
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Streamlit
- NumPy, Pandas, Matplotlib

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your browser at http://localhost:8501

## Training an Agent

You can train your own RL agent using the training script:

```
python training.py
```

or use the Training interface in the Streamlit app.

## How It Works

### Reinforcement Learning Approach

The DQN agent learns optimal poker strategies through:

1. **State Representation**: Cards, pot size, betting history
2. **Action Space**: Fold, check, call, bet, raise
3. **Reward Function**: Change in chip stack
4. **Neural Network**: Multi-layer perceptron for Q-value estimation
5. **Experience Replay**: Learning from past gameplay

### Game Mechanics

The poker implementation follows Texas Hold'em rules with:

- Betting rounds: preflop, flop, turn, river
- Hand evaluation for showdowns
- Proper betting mechanics (raise, call, check, fold)

## Future Enhancements

- Implement more sophisticated RL algorithms (A2C, PPO)
- Add hand strength calculation and pot odds
- Improve state representation for better learning
- Multi-agent training with population-based methods
- Advanced poker variants

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI for reinforcement learning research
- PyTorch team for the deep learning framework
- Streamlit for the interactive UI capabilities
