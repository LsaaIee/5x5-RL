import csv
import random
from agents import QLearningAgent, RandomAgent, NoisyHeuristicAgent, AlphaBetaAgent
from game import RL_TicTacToe_Env

def evaluate_policy(model_path, opponent_type, total_games):
    q_agent = QLearningAgent()
    q_agent.load_model(model_path)

    epsilon = 0.0

    wins = 0
    losses = 0
    draws = 0
    total_return = 0
    total_moves = 0

    for i in range(total_games):
        random.seed(i)

        if i < (total_games/2):
            q_mark = 'X'
        else:
            q_mark = 'O'

        if opponent_type == "Random":
            opponent = RandomAgent()
        elif opponent_type == "Noisy": 
            opponent = NoisyHeuristicAgent()
        elif opponent_type == "AlphaBeta":
            opponent = AlphaBetaAgent()
        
        env = RL_TicTacToe_Env(opponent, QAgentMark=q_mark)
        current_state = env.reset()
        done = False
        game_reward = 0

        while done == False:
            legal_actions = env.get_legal_actions()
            action = q_agent.select_action(current_state, legal_actions, epsilon)

            if i == 0:
                q_value = q_agent.get_q_value(current_state, action)
                # Keep your original single line print just in case
                # print(f"[Game 0] Agent {q_mark} chooses action {action} with Q-value: {q_value:.4f}")

            next_state, reward, done = env.step(action)
            current_state = next_state
            game_reward = game_reward+reward

        if game_reward > 0:
            wins += 1
        elif game_reward < 0:
            losses += 1
        else:
            draws += 1

        total_return = total_return+game_reward
        total_moves = total_moves + (25-env.game.empty_cells)
    
    print(f"\n--- Evaluation: vs {opponent_type} Summary ---")
    print(f"Wins: {wins} / Losses: {losses} / Draws: {draws}")

    results_data = {
        "opponent": opponent_type,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_return": total_return/total_games,
        "avg_game_length": total_moves/total_games
    }

    return results_data

def write_to_csv_file(filename, rows):
    if not rows:
        return
    headers = rows[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def visualize_game(env, q_agent, agent_mark='X', show_heatmap=False):
    print("\n--- STARTING VISUALIZATION GAME ---")
    state = env.reset()
    print("Initial Board State:")
    env.print_board() 
    
    done = False
    while not done:
        legal_moves = env.game.get_legal_moves()
        
        action = q_agent.select_action(state, legal_moves, epsilon=0.0)
        q_value = q_agent.get_q_value(state, action)
        
        print(f"\nPlayer {agent_mark} (Q-Agent) plays {action}")
        print(f"--> Expected Q-value for this move: {q_value:.3f}")
        
        if show_heatmap:
            print("   [Q-Value Heatmap for Legal Moves]")
            for move in legal_moves:
                val = q_agent.get_q_value(state, move)
                print(f"    Move {move}: {val:.3f}")
            print("-------------------------------------")
            show_heatmap = False 
            
        next_state, reward, done = env.step(action)
        
        print("Board state after turn:")
        env.print_board() 
        state = next_state
        
    print("--- GAME OVER ---")
    if reward == 1.0:
        print("Result: Q-Agent Wins!")
    elif reward == -1.0:
        print("Result: Opponent Wins.")
    else:
        print("Result: Draw.")

if __name__ == "__main__":
    csv_rows = []
    visualize = False

    print("Evaluating Baseline (Config 1)...")
    temp_agent1 = QLearningAgent()
    temp_agent1.load_model("q_table_baseline.pkl")
    baseline_q_size = len(temp_agent1.q_table)

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
            "TrainingTime": "Reported in train.py"
        }
        csv_rows.append(row)

    print("\nEvaluating Improved (Config 2)...")
    temp_agent2 = QLearningAgent()
    temp_agent2.load_model("q_table_improved.pkl")
    improved_q_size = len(temp_agent2.q_table)

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
            "TrainingTime": "Reported in train.py"
        }
        csv_rows.append(row)

    write_to_csv_file("results.csv", csv_rows)
    print("\nresults.csv generated successfully.")
    
    if visualize == True:
        print("\nRunning full visualization of all evaluation games (Baseline & Improved)...")
        # List of both agents to iterate through
        agents_to_visualize = [
            ("Baseline (Config 1)", temp_agent1),
            ("Improved (Config 2)", temp_agent2)
        ]

        for config_name, current_agent in agents_to_visualize:
            print(f"\n=================================================")
            print(f"VISUALIZING: {config_name.upper()}")
            print(f"=================================================")
            
            for opponent_type in ["Random", "Noisy", "AlphaBeta"]:
                total_games = 20 if opponent_type == "AlphaBeta" else 100
                
                if opponent_type == "Random":
                    vis_opponent = RandomAgent()
                elif opponent_type == "Noisy":
                    vis_opponent = NoisyHeuristicAgent() 
                elif opponent_type == "AlphaBeta":
                    vis_opponent = AlphaBetaAgent() 
                    
                vis_env = RL_TicTacToe_Env(vis_opponent, 'X')
                
                print(f"\n\n>>> VISUALIZING GAMES AGAINST: {opponent_type.upper()} <<<")
                
                for i in range(total_games):
                    print(f"\n=== {config_name} | Game {i + 1} of {total_games} vs {opponent_type} ===")
                    
                    # Show heatmap ONLY for the very first game against the Random Agent
                    if opponent_type == "Random" and i == 0:
                        visualize_game(vis_env, current_agent, agent_mark='X', show_heatmap=True)
                    else:
                        visualize_game(vis_env, current_agent, agent_mark='X', show_heatmap=False)