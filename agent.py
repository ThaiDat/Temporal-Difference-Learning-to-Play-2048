from globalconfig import gconfig
import numpy as np


class NTuple:
    '''Most basic element of weightless network'''
    def __init__(self, pattern, chanel_encoded=gconfig['CHANEL_ENCODED']):
        '''
        pattern: list of position (row, col) for pattern recognizer
        chanel_encoded: Length of one hot encoded vector of each tile
        '''
        self.pattern = pattern
        self.table = np.zeros((chanel_encoded,) * len(pattern), dtype=np.float32)

    def look_up(self, board):
        '''
        Rotate and flip pattern to assess board
        board: pure board representation
        return board value function
        '''
        boards = self.__flip_rotate_board(self.__preprocess_board(board))
        v = 0
        for board in boards:
            indices =tuple(board[r, c] for r, c in self.pattern)
            v += self.table[indices]
        return v
            
    def update(self, board, update_value):
        '''
        Update assessment for given board
        board: pure board representation
        '''
        boards = self.__flip_rotate_board(self.__preprocess_board(board))
        update_value /= 8
        for board in boards:
            indices =tuple(board[r, c] for r, c in self.pattern)
            self.table[indices] += update_value
    
    def __preprocess_board(self, board):
        '''
        Convert pure board representation to board of indices
        board: pure board representation
        return np array board of indices
        '''
        board = np.array(board)
        board[board==0] = 1
        board = np.log2(board).astype(np.int32)
        return board

    def __flip_rotate_board(self, board):
        '''
        Do the flip and rotation on board to get the same effect of pattern
        board: numpy array board
        return list of board variations
        '''
        board_flip = board[:, ::-1]
        variations = [
            board,
            np.rot90(board, 1, (0, 1)),
            np.rot90(board, 2, (0, 1)),
            np.rot90(board, 3, (0, 1)),
            board_flip,
            np.rot90(board_flip, 1, (0, 1)),
            np.rot90(board_flip, 2, (0, 1)),
            np.rot90(board_flip, 3, (0, 1)),
        ]
        return variations


class WeightlessNetworkModel:
    '''Model of value function using weightless network'''
    def __init__(self, patterns=gconfig['NTUPLES'], lr=gconfig['LEARNING_RATE']):
        '''
        patterns: matrix of N x ntuples. Define how to create ntuples
        '''        
        self.patterns = [NTuple(pattern) for pattern in patterns]
        self.lr = lr

    def predict(self, states):
        '''
        Predict values of given states
        states: list of pure board representation
        return values of corresponding states
        '''
        return [np.sum([p.look_up(state) for p in self.patterns]) for state in states]

    def fit(self, states, values):
        '''
        Train the network with given datas
        states: list of pure board representations
        values: list of state values
        return mean loss
        '''
        current_values = self.predict(states)
        updates = np.subtract(values, current_values)
        self.fit_update(states, updates)
        return np.mean(np.square(updates))
        
    def fit_update(self, states, updates):
        '''
        Train the network with pre-calculated updates
        states: list of pure board representations
        updates: list of directions of each update. Normally old_value -> new_value
        '''
        for state, update in zip(states, updates):
            update = update * self.lr / len(self.patterns)
            for pattern in self.patterns:
                pattern.update(state, update)
            
            
