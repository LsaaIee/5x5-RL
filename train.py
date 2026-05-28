import random
import time
import json
from agents import QLearningAgent, RandomAgent, NoisyHeuristicAgent
from game import RL_TicTacToe_Env

def train_agent(configuration_type):
    random.seed(15)
    num_epis = 500000
    alpha = 0.1
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05

    q_agent = QLearningAgent(alpha, gamma)
    episode_returns = []

    for episode in range(num_epis):
        current_epsilon = calculate_decay(episode, epsilon_start, epsilon_end, num_epis)

        if configuration_type == "Baseline (Config 1)":
            opponent = RandomAgent()
        elif configuration_type == "Improved (Config 2)": 
            if random.random() < 0.5:
                opponent = RandomAgent()
            else: 
                opponent = NoisyHeuristicAgent()
        
        mark = random.choice(['X', 'O'])
        env = RL_TicTacToe_Env(opponent, QAgentMark=mark)

        current_state = env.reset()
        total_episode_reward = 0
        done = False

        while done == False:
            legal_actions = env.get_legal_actions()
            action = q_agent.select_action(current_state, legal_actions, current_epsilon)

            next_state, reward, done = env.step(action)

            if configuration_type == "Improved (Config 2)":
                reward = apply_custom_reward_shaping(reward, done)
            
            next_legal_actions = env.get_legal_actions()

            q_agent.update(current_state, action, reward, next_state, next_legal_actions, done)

            current_state = next_state
            total_episode_reward = total_episode_reward+reward

        episode_returns.append(total_episode_reward)

    if configuration_type == "Baseline (Config 1)":
        q_agent.save_model("q_table_baseline.pkl")
    else:
        q_agent.save_model("q_table_improved.pkl")

    return episode_returns

def calculate_decay(episode, epsilon_start, epsilon_end, total_episodes):
    progress = episode/total_episodes
    epsilon = epsilon_start-(progress*(epsilon_start-epsilon_end))

    return max(epsilon, epsilon_end)

def apply_custom_reward_shaping(base_reward, done):
    if done == True:
        return base_reward
    else:
        step_penalty = -0.01
        return base_reward+step_penalty

if __name__ == "__main__":
    print("Starting Training for Config 1 (Baseline)...")
    start_time = time.time()
    baseline_returns = train_agent("Baseline (Config 1)")
    baseline_time = time.time() - start_time
    print(f"Config 1 trained in: {baseline_time:.2f} seconds")

    with open("baseline_returns.json", "w") as f:
        json.dump(baseline_returns, f)

    print("\nStarting Training for Config 2 (Improved)...")
    start_time = time.time()
    improved_returns = train_agent("Improved (Config 2)")
    improved_time = time.time() - start_time
    print(f"Config 2 trained in: {improved_time:.2f} seconds")

    with open("improved_returns.json", "w") as f:
        json.dump(improved_returns, f)
        
    print("\nTraining complete! 'q_table_baseline.pkl' and 'q_table_improved.pkl' have been saved.")