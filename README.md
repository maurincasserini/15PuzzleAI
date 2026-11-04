# 15-Puzzle Solver with REINFORCE (PyTorch)

A policy-gradient agent that learns to solve the 15-puzzle from scratch using the REINFORCE algorithm in PyTorch without external RL frameworks.

[Watch the trained agent (mp4)](15Puzzle_demo.mp4)

## Overview

The agent observes the 16-tile board state encoded as a one-hot vector, along with the previous action taken, and outputs a discrete move choice (up, down, left, right). The reward structure encourages reducing the total Manhattan distance of the tiles while penalizing state loops and step count.

## Method

- **Algorithm:** REINFORCE (Monte-Carlo policy gradient), updated in mini-batches every 50 episodes
- **Policy:** Categorical distribution over valid discrete actions; invalid actions are masked with high negative logits ($-10^{10}$)
- **Network:** MLP `260 → 256 → 256 → 128 → 4` with ReLU activations
- **State Representation:** 256-dimensional one-hot encoded board tensor + 4-dimensional previous action vector ($260$ inputs total)
- **Reward Shaping:**
  - **Manhattan Distance:** Positive reward is awarded only when the total Manhattan distance decreases relative to the previous step; otherwise, no positive reward is given.
  - **Loop Penalty:** Penalises visiting any state already seen earlier in the same episode.
  - **Step Penalty:** Constant penalty ($-2$) per move to encourage faster solutions.
  - **Goal:** $+200$ reward upon solving the puzzle.
- **Returns:** Discounted with $\gamma = 0.98$, standardized per batch (zero mean, unit std) to stabilize training
- **Loss:** $-\sum \log \pi(a \mid s) \cdot G_t$
- **Optimizer:** Adam, learning rate $10^{-3}$, gradient norm clipping at $1.0$
- **Checkpointing:** Weights saved to `15Puzzle.pth` every 10,000 episodes

## Project structure

```
.
├── 15PuzzleRLManhattanDistance.py        # Environment logic, network architecture, REINFORCE training loop
├── 15PuzzleRLManhattanDistanceTest.py         # Evaluation script to run the trained policy
├── 15Puzzle.pth    # Saved model weights
├── 15Puzzle_demo.mp4        # Video recording of the trained agent
└── README.md
```


## Key Learnings & Debugging Insights

- **The Oscillation Problem:** Initially, the neural network got stuck in local minima, simply shuffling the zero tile back and forth at the bottom of the board.
- **First Failed Fix (Action Memory):** Passing the previous step into the network input and penalizing immediate reverse actions did not solve the issue; the agent simply expanded its loop pattern and started moving the zero tile in circles.
- **The Solution (History-Based Penalty):** Introducing a penalty for revisiting *any* state already seen in the episode fixed the behavior. Even though the network receives no direct input history of visited states, it learned that the only way to consistently avoid this negative reward was to develop a structured, goal-driven solving strategy.

## Results

The AI is a slow learner and it probably isn't the best solution although it works out most 15 Puzzles even if it isn't the fastest solution.
