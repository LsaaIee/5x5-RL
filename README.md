# 5x5 Tic-Tac-Toe RL version

## Overview
This project implements a Tabular Q-Learning agent to play a 5x5 variant of Tic-Tac-Toe. The agent formulates the game as a Markov Decision Process (MDP) and learns optimal policies through offline training against baseline opponents (Random and Noisy Heuristic). The project evaluates two configurations: a Baseline using strictly terminal rewards, and an Improved Configuration utilizing a custom step penalty.

## Project Structure & Files
* **`agents.py`**: Contains the logic for the `QLearningAgent` (including the Bellman update, epsilon-greedy selection, and save/load functions) alongside the HW1 opponent agents.
* **`game.py`**: Contains the base `TicTacToe5x5` logic and the `RL_TicTacToe_Env` wrapper, which converts the adversarial game into a single-player MDP by automatically absorbing opponent moves.
* **`train.py`**: Handles the episodic training loop, epsilon decay, reward shaping implementation, and Q-table serialization (`.pkl`). Also, returns the reward of each game. 
* **`evaluate.py`**: Runs strict, 100% greedy ($\epsilon=0.0$) evaluation matches with reproducible seeds and fair 50/50 starting player distributions. Also, it visualizes board states of every game after each move. 

## How to Run
This project relies entirely on standard Python libraries (`random`, `time`, `math`, `pickle`, `csv`, `json`, `copy`). No external dependencies are required. Therefore, `requirements.txt` is an empty file. 

To execute the train session, run: 
```bash
python train.py
```
This will train the agent. After training is complete, respective `.pkl` and `.json` files will be generated. 

To start evaluation with the generated `.pkl` files, run:
```bash
python evaluate.py
```
This will evaluate the result, and output a `results.csv` summary.

To activate turn-by-turn visualization, have to change line 124 of `evaluate.py` from
```python
visualize = False
```
to 
```python
visualize = True
``` 
and run:
```bash
python evaluate.py > visualized.txt
```
to avoid terminal buffer overflow(part of result missing problem at the terminal)

The `visualized.txt` will show the overview of the result, the heatmap of the first game only, and turn-by-turn visualizations total of 440 games. 
