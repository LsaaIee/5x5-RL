import copy
import random

class TicTacToe5x5:
    def __init__(self):
        # Initialize a 5x5 board with empty spaces.
        # Board cells are indexed (0,0) to (4,4) as per specifications.
        self.board = [[' ' for _ in range(5)] for _ in range(5)]
        self.current_player = 'X' # X always goes first logically, but actual starting player is random per game
        self.empty_cells = 25

    def get_legal_moves(self):
        """Returns a list of (row, col) tuples representing empty cells."""
        moves = []
        for r in range(5):
            for c in range(5):
                if self.board[r][c] == ' ':
                    moves.append((r, c))
        return moves

    def make_move(self, move, player):
        """Applies a move to the board and returns a new game state (useful for search trees)."""
        r, c = move
        if self.board[r][c] != ' ':
            raise ValueError("Invalid move: Cell is already occupied.")
        
        # We return a new instance so the Alpha-Beta search doesn't overwrite the actual game board
        new_state = copy.deepcopy(self)
        new_state.board[r][c] = player
        new_state.empty_cells -= 1
        new_state.current_player = 'O' if player == 'X' else 'X'
        return new_state

    def check_win(self, player):
        """Checks if the specified player has 4 consecutive marks in a row, col, or diagonal."""
        b = self.board
        
        # Check horizontal and vertical
        for i in range(5):
            for j in range(2): # We only need to check starting at col/row 0 and 1 for a length of 4
                # Horizontal
                if b[i][j] == player and b[i][j+1] == player and b[i][j+2] == player and b[i][j+3] == player:
                    return True
                # Vertical
                if b[j][i] == player and b[j+1][i] == player and b[j+2][i] == player and b[j+3][i] == player:
                    return True

        # Check diagonals
        for i in range(2): # Row starting points: 0, 1
            for j in range(2): # Col starting points: 0, 1
                # Top-left to bottom-right
                if b[i][j] == player and b[i+1][j+1] == player and b[i+2][j+2] == player and b[i+3][j+3] == player:
                    return True
                # Top-right to bottom-left (starting from right side: cols 4, 3)
                if b[i][4-j] == player and b[i+1][3-j] == player and b[i+2][2-j] == player and b[i+3][1-j] == player:
                    return True

        return False

    def is_terminal(self):
        """Returns True if the game is over (win or draw), else False."""
        if self.check_win('X') or self.check_win('O'):
            return True
        if self.empty_cells == 0: # Draw condition: all 25 cells filled with no winner
            return True
        return False

    def print_board(self):
        """A simple text-based display to meet the visualization requirement."""
        print("  0 1 2 3 4")
        for i, row in enumerate(self.board):
            print(f"{i} " + "|".join(row))
            if i < 4:
                print("  " + "-" * 9)
        print()

class RL_TicTacToe_Env:
    def __init__(self, OpponentAgent, QAgentMark):
        self.game = TicTacToe5x5()
        self.opponent = OpponentAgent
        self.q_mark = QAgentMark
        self.opp_mark = 'X' if QAgentMark == 'O' else 'O'

    def get_state(self):
        board_1D = []
        for row in self.game.board:
            for cell in row:
                board_1D.append(cell)
        
        return tuple(board_1D)
    
    def get_legal_actions(self):
        return self.game.get_legal_moves()
    
    def reset(self):
        self.game = TicTacToe5x5()
        starting_player = random.choice(['X', 'O'])

        if starting_player == self.opp_mark:
            opp_action = self.opponent.get_move(self.game, self.opp_mark, self.q_mark)
            r, c = opp_action
            self.game.board[r][c] = self.opp_mark
            self.game.empty_cells -= 1
        
        return self.get_state()

    def step(self, q_action):
        r, c = q_action
        self.game.board[r][c] = self.q_mark
        self.game.empty_cells -= 1

        if self.game.check_win(self.q_mark) == True:
            return self.get_state(), 1.0, True
        if self.game.is_terminal() == True:
            return self.get_state(), 0.0, True

        opp_action = self.opponent.get_move(self.game, self.opp_mark, self.q_mark)
        r_opp, c_opp = opp_action
        self.game.board[r_opp][c_opp] = self.opp_mark
        self.game.empty_cells -= 1

        if self.game.check_win(self.opp_mark) == True:
            return self.get_state(), -1.0, True
        if self.game.is_terminal() == True:
            return self.get_state(), 0.0, True
        
        return self.get_state(), 0.0, False
    
    def print_board(self):
        self.game.print_board()