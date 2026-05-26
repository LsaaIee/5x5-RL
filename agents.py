import random
import time
import math
import pickle

class RandomAgent: 
    def get_move(self, game_state, current_player, opponent_player):
        # Retrieve all available empty cells on the 5x5 board
        legal_moves = game_state.get_legal_moves()
        
        # Safety check: ensure the game isn't already over
        if len(legal_moves) == 0: 
            raise ValueError("No moves available, board is full.")
        
        return random.choice(legal_moves)
    
class NoisyHeuristicAgent:
    def get_move(self, game_state, current_player, opponent_player):
        legal_moves = game_state.get_legal_moves()

        # 1. The "noise" check
        # generate a random float between 0.0 and 1.0
        if random.random() < 0.2:
            return random.choice(legal_moves)
        
        # 2. the heuristic logic
        # rule A: immediate win detection
        for move in legal_moves:
            simulated_state = game_state.make_move(move, current_player)
            if simulated_state.check_win(current_player):
                return move
            
        # rule B: block opponent's win
        for move in legal_moves:
            simulated_state = game_state.make_move(move, opponent_player)
            if simulated_state.check_win(opponent_player):
                return move
            
        # rule C: positional preference (center)
        center_move = (2, 2)
        if center_move in legal_moves:
            return center_move
        
        #rule D: fallback
        return random.choice(legal_moves)
    
class AlphaBetaAgent: 
    depth_limit = 3
    max_time = 2.9

    def get_move(self, game_state, current_player, opponent_player):
        self.nodes_expanded = 0
        start_time = time.time()
        legal_moves = game_state.get_legal_moves()

        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for move in legal_moves:
            if (time.time() - start_time) > self.max_time: 
                return random.choice(legal_moves)
            
            self.nodes_expanded += 1                
            simulated_state = game_state.make_move(move, current_player)
            value = self.alpha_beta(simulated_state, self.depth_limit-1, alpha, beta, False, current_player, opponent_player)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move
    
    def alpha_beta(self, state, depth, alpha, beta, is_maximizing, my_player, opp_player):
        if depth == 0 or state.is_terminal():
            return self.evaluate(state, my_player, opp_player)
        
        legal_moves = state.get_legal_moves()

        if is_maximizing:
            max_eval = -math.inf
            for move in legal_moves:
                self.nodes_expanded += 1
                sim_state = state.make_move(move, my_player)
                eval = self.alpha_beta(sim_state, depth-1, alpha, beta, False, my_player, opp_player)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    break
                
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                self.nodes_expanded += 1
                sim_state = state.make_move(move, opp_player)
                eval = self.alpha_beta(sim_state, depth-1, alpha, beta, True, my_player, opp_player)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                if beta <= alpha:
                    break
                
            return min_eval
    
    def score_line(self, line_array, my_player, opp_player):
        line_str = "".join(line_array)

        if my_player in line_str and opp_player in line_str:
            return 0

        my_count = line_str.count(my_player)
        opp_count = line_str.count(opp_player)

        score = 0

        if my_count == 4:
            score += 100000
        elif my_count == 3:
            if line_str == f" {my_player}{my_player}{my_player} ":
                score += 50000
            else:
                score += 50
        elif my_count == 2:
            score += 10
        
        if opp_count == 4:
            score -= 100000
        elif opp_count == 3:
            if line_str == f" {opp_player}{opp_player}{opp_player} ":
                score -= 50000
            else:
                score -= 50
        elif opp_count == 2:
            score -= 10
        
        return score

    def evaluate(self, state, my_player, opp_player):
        if state.check_win(my_player):
            return 10000000  # Alapha-Beta wins
        elif state.check_win(opp_player):
            return -10000000  # Noisy/Random wins
        elif state.is_terminal():
            return 0  # Draw
        
        lines = []

        for r in range(5):
            row_line = state.board[r]
            lines.append(row_line)
        
        for c in range(5):
            col_line = [state.board[r][c] for r in range(5)]
            lines.append(col_line)

        diag1 = [state.board[i][i] for i in range(5)]
        diag2 = [state.board[i][4-i] for i in range(5)]
        diag3 = [state.board[i][i+1] for i in range(4)]
        diag4 = [state.board[i+1][i] for i in range(4)]
        diag5 = [state.board[i][3-i] for i in range(4)]
        diag6 = [state.board[i+1][4-i] for i in range(4)]
        lines.extend([diag1, diag2, diag3, diag4, diag5, diag6])

        score = 0
        for line in lines:
            score += self.score_line(line, my_player, opp_player)

        return score
    
class QLearningAgent: 
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
    
    def get_q_value(self, state, action):
        "sparse q-table: if state-action pair not found from q-table, init as 0.0"
        if (state, action) in self.q_table: 
            return self.q_table[(state, action)]
        else: 
            return 0.0
        
    def select_action(self, state, legal_actions, epsilon):
        "select the next move among equally best actions"
        randnum = random.random()

        if randnum < epsilon:
            return random.choice(legal_actions)
        else: 
            max_q = -math.inf
            best_actions = []

            for action in legal_actions:
                current_q = self.get_q_value(state, action)

                if current_q > max_q:
                    max_q = current_q
                    best_actions = [action]
                elif current_q == max_q:
                    best_actions.append(action)
            
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.get_q_value(state, action)

        if done == True:
            target = reward
        else:
            max_next_q = -math.inf

            for next_action in next_legal_actions:
                next_q_val = self.get_q_value(next_state, next_action)
                if next_q_val > max_next_q:
                    max_next_q = next_q_val
            
            if max_next_q == -math.inf:
                max_next_q = 0.0
            
            target = reward + self.gamma * max_next_q
        
        # Q-value update 
        new_q = current_q + self.alpha * (target-current_q)
        self.q_table[(state, action)] = new_q

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)