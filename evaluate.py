from agents import QLearningAgent, RandomAgent, NoisyHeuristicAgent, AlphaBetaAgent
import random
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
                print(f"[Game 0] Agent {q_mark} chooses action {action} with Q-value: {q_value:.4f}")

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