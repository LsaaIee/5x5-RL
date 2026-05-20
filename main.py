from game import TicTacToe5x5
from agents import RandomAgent, NoisyHeuristicAgent, AlphaBetaAgent
import time
import random

def play_single_game(agent_x, agent_o, visualize=False): 
    board = TicTacToe5x5()
    current_player = 'X'
    start_time = time.time()
    move_cnt = 0
    nodes_expanded = 0

    while not board.is_terminal(): 
        if current_player == 'X':
            move = agent_x.get_move(board, 'X', 'O')
            if hasattr(agent_x, 'nodes_expanded'):
                nodes_expanded += agent_x.nodes_expanded
        else: 
            move = agent_o.get_move(board, 'O', 'X')
            if hasattr(agent_o, 'nodes_expanded'):
                nodes_expanded += agent_o.nodes_expanded

        board = board.make_move(move, current_player)
        move_cnt += 1

        if visualize == True:
            print(f"Player {current_player} plays {move}")
            board.print_board()
        
        if current_player == 'X':
            current_player = 'O' 
        else: 
            current_player = 'X'
        
    end_time = time.time()
    total_time = end_time-start_time

    if board.check_win('X'):
        winner = 'X'
    elif board.check_win('O'):
        winner = 'O'
    else: 
        winner = 'Draw'

    if visualize:
        print(f"Final result: {winner}")
        print(f"Computation Time: {total_time}")
        print(f"Total Moves: {move_cnt}")

    return winner, total_time, nodes_expanded, move_cnt

def run_all_experiments():
    agent_R = RandomAgent()
    agent_N = NoisyHeuristicAgent()
    agent_AB = AlphaBetaAgent()

    # Experiment against Noisy agent
    opponent = agent_N
    winners = []
    times = []
    moves_list = []
    nodes_list = []

    print("--- Starting 20 Games vs Noisy Agent ---")
    for i in range(20):
        random.seed(i)

        if random.choice([True, False]):
            agent_x = agent_AB
            agent_o = opponent
            ab_role = 'X'
        else:
            agent_x = opponent
            agent_o = agent_AB
            ab_role = 'O'

        winner, time_taken, nodes, moves = play_single_game(agent_x, agent_o, visualize=False)

        if winner == ab_role:
            winners.append('Alpha-Beta')
        elif winner == 'Draw':
            winners.append('Draw')
        else: 
            winners.append('Opponent')
        times.append(time_taken)
        moves_list.append(moves)
        nodes_list.append(nodes)
    
    print("--- Final Experiment Summary(vs. Noisy) ---")
    print(f"Player X (Alpha-Beta) Wins: {winners.count('Alpha-Beta')}")
    print(f"Player O (the opponent) Wins: {winners.count('Opponent')}")
    print(f"Draws: {winners.count('Draw')}")
    print(f"Average Moves: {sum(moves_list)/len(moves_list)}")
    print(f"Average Nodes Expanded: {sum(nodes_list)/len(nodes_list)}")

    # Experiment against Random agent
    opponent = agent_R
    winners = []
    times = []
    moves_list = []
    nodes_list = []

    print("\n--- Starting 20 Games vs Random Agent ---")
    for i in range(20):
        random.seed(i)

        if random.choice([True, False]):
            agent_x = agent_AB
            agent_o = opponent
            ab_role = 'X'
        else:
            agent_x = opponent
            agent_o = agent_AB
            ab_role = 'O'

        winner, time_taken, nodes, moves = play_single_game(agent_x, agent_o, visualize=False)

        if winner == ab_role:
            winners.append('Alpha-Beta')
        elif winner == 'Draw':
            winners.append('Draw')
        else: 
            winners.append('Opponent')
        times.append(time_taken)
        moves_list.append(moves)
        nodes_list.append(nodes)

    print("--- Final Experiment Summary(vs. Random) ---")
    print(f"Player X (Alpha-Beta) Wins: {winners.count('Alpha-Beta')}")
    print(f"Player O (the opponent) Wins: {winners.count('Opponent')}")
    print(f"Draws: {winners.count('Draw')}")
    print(f"Average Moves: {sum(moves_list)/len(moves_list)}")
    print(f"Average Nodes Expanded: {sum(nodes_list)/len(nodes_list)}")

if __name__ == "__main__":
    run_all_experiments()