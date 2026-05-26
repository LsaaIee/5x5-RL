# 5x5 Tic-Tac-Toe RL version

## Overview
This project implements a Tabular Q-Learning agent to play a 5x5 variant of Tic-Tac-Toe. The agent formulates the game as a Markov Decision Process (MDP) and learns optimal policies through offline training against baseline opponents (Random and Noisy Heuristic). The project evaluates two configurations: a Baseline using strictly terminal rewards, and an Improved Configuration utilizing a custom step penalty.

## Project Structure & Files
* **`main.py`**: The orchestrator script. Automates the training of both configurations, triggers the evaluation loop, generates the `results.csv` file, and handles the final visualization game.
* **`agents.py`**: Contains the logic for the `QLearningAgent` (including the Bellman update, epsilon-greedy selection, and save/load functions) alongside the HW1 opponent agents.
* **`game.py`**: Contains the base `TicTacToe5x5` logic and the `RL_TicTacToe_Env` wrapper, which converts the adversarial game into a single-player MDP by automatically absorbing opponent moves.
* **`train.py`**: Handles the episodic training loop, epsilon decay, reward shaping implementation, and Q-table serialization (`.pkl`).
* **`evaluate.py`**: Runs strict, 100% greedy ($\epsilon=0.0$) evaluation matches with reproducible seeds and fair 50/50 starting player distributions.

## How to Run the Code
This project relies entirely on standard Python libraries (`random`, `time`, `math`, `pickle`, `csv`, `json`, `copy`). No external dependencies are required. Therefore, `requirements.txt` is an empty file. 

To execute the full training and evaluation pipeline, run:
```bash
python main.py
```
This will train the agent, evaluate the result, and output a `results.csv` and the respective `.json` and `.pkl` files. 

To activate visualization, have to change line 139 of main.py from
```python
run_RL(visualize=False)
```
to 
```python
run_RL(visualize=True)
``` 

To avoid terminal buffer overflow(part of result missing problem at the terminal), run:
```bash
python main.py > every_game_progress.txt
```
to see the entire game result with visualization. 