import numpy as np
from math import inf as infinity


class RandomConnect4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanConnect4Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)

        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                print('Invalid move')
        return move


class GreedyConnect4Player():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, verbose=False):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid:
                continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose:
                print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose:
                print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose:
                print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % self.game.stringRepresentation(board))

        return ret_move


class MinMaxConnect4Player():
    def __init__(self, game, depth):
        self.game = game
        self.depth = depth

    def play(self, board):
        score = self.minimax((board, -1), self.depth, 1, -infinity, +infinity)
        return score[0]

    def minimax(self, state, depth, player, alfa, beta):

        best = [None, None]

        if player == 1:
            best[1] = -infinity
        else:
            best[1] = +infinity

        '''
        if depth==0 or self.game.getGameEnded(state[0],player)!=0:
            score=self.game.getGameEnded(state[0],player)
            return [None,score]
        '''

        if self.game.getGameEnded(state[0],player) != 0:
            score = self.game.getGameEnded(state[0], player)
            return [None, score+depth/(depth+1)*-player]
        elif depth == 0:
            score = self.game.getScore(state[0], player)
            return [None, score]

        valids = self.game.getValidMoves(state[0], player)
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard= self.game.getNextState(state[0], player, a)
            score = self.minimax(nextBoard, depth-1, -player, alfa, beta)

            if player == 1:
                if score[1] > best[1]:
                    best[1] = score[1]
                    best[0] = a
                alfa = max(alfa, best[1])
                if beta <= alfa:
                    break
            else:
                if score[1] < best[1]:
                    best[1] = score[1]
                    best[0] = a
                beta = min(beta, best[1])
                if beta <= alfa:
                    break
        return best
