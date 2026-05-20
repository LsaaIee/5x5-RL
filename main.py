import time
import csv
import json
import random
from train import train_agent
from evaluate import evaluate_policy
from game import TicTacToe5x5
from agents import RandomAgent, NoisyHeuristicAgent, AlphaBetaAgent, QLearningAgent

def write_to_csv_file(filename, rows):
    if not rows:
        return
    headers = rows[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def run_RL():
    csv_rows = []

    print("Starting Training for Config 1...")
    start_time = time.time()
    baseline_return = train_agent("Baseline (Config 1)")
    end_time = time.time()
    baseline_training_time = end_time-start_time
    print(f"Config 1 trained in: {baseline_training_time:.2f} seconds")

    temp_agent1 = QLearningAgent()
    temp_agent1.load_model("q_table_baseline.pkl")
    baseline_q_size = len(temp_agent1.q_table)
    print(f"Baseline Q-table Size: {baseline_q_size} entries")

    with open("baseline_returns.json", "w") as f:
        json.dump(baseline_return, f)

    for opponent_type in ["Random", "Noisy", "AlphaBeta"]:
        total_games = 20 if opponent_type == "AlphaBeta" else 100
        metrics = evaluate_policy("q_table_baseline.pkl", opponent_type, total_games)

        row = {
            "Config": "Baseline (Config 1)",
            "Opponent": opponent_type,
            "Wins": metrics["wins"],
            "Losses": metrics["losses"],
            "Draws": metrics["draws"],
            "AvgReturn": metrics["avg_return"],
            "AvgGameLength": metrics["avg_game_length"],
            "QTableSize": baseline_q_size,
            "TrainingTime": baseline_training_time
        }
        csv_rows.append(row)

    print("\nStarting Training for Config 2...")
    start_time = time.time()
    improved_return = train_agent("Improved (Config 2)")
    end_time = time.time()
    improved_training_time = end_time-start_time
    print(f"Config 2 trained in: {improved_training_time:.2f} seconds")

    temp_agent2 = QLearningAgent()
    temp_agent2.load_model("q_table_improved.pkl")
    improved_q_size = len(temp_agent2.q_table)
    print(f"Improved Q-table Size: {improved_q_size} entries")

    with open("improved_returns.json", "w") as f:
        json.dump(improved_return, f)

    for opponent_type in ["Random", "Noisy", "AlphaBeta"]:
        total_games = 20 if opponent_type == "AlphaBeta" else 100
        metrics = evaluate_policy("q_table_improved.pkl", opponent_type, total_games)

        row = {
            "Config": "Improved (Config 2)",
            "Opponent": opponent_type,
            "Wins": metrics["wins"],
            "Losses": metrics["losses"],
            "Draws": metrics["draws"],
            "AvgReturn": metrics["avg_return"],
            "AvgGameLength": metrics["avg_game_length"],
            "QTableSize": improved_q_size,
            "TrainingTime": improved_training_time
        }
        csv_rows.append(row)
    
    write_to_csv_file("results.csv", csv_rows)
    print("\nresults.csv generated. All Done")    

if __name__ == "__main__":
    run_RL()