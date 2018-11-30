import numpy as np
from tictactoe.TicTacToeLogic import Board

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""
class RandomTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)

        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyTicTacToePlayer():
    def __init__(self,game):
        self.game=game

    def play(self,board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            move = (int(a / self.game.n), a% self.game.n)
           # print(str(score)+" "+str(move))
            candidates += [(-score, a)]
        candidates.sort()
        print(str(candidates))
        return candidates[0][1]


class MinMaxTicTacToePlayer():
    def __init__(self,game):
       self.game=game

    def play(self,board):
        valids=self.game.getValidMoves(board,1)
        candidates=[]
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard=self.game.getNextState(board,1,a)
            score = self.minimax(nextBoard,9,-1)
            candidates+=[(score[1],a)]
        candidates.sort()
        return candidates[0][1]

    def minimax(self,state,depth,player):

        best = [None, None]

        if player==1:
            best[1]=-1
        else:
            best[1]=1


        if depth==0 or self.game.getGameEnded(self.game.getCanonicalForm(state[0],player),player)!=0:
            score=self.game.getScore(self.game.getCanonicalForm(state[0],player),player)
            return [None,score]

        valids = self.game.getValidMoves(self.game.getCanonicalForm(state[0],player), player)
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard= self.game.getNextState(self.game.getCanonicalForm(state[0],player), player, a)
            score = self.minimax(nextBoard, depth-1, -player)

            if player==1:
                if score[1] > best[1]:
                    best[1]=score[1]
                    best[0]=a
            else:
                if score[1]<best[1]:
                    best[1]=score[1]
                    best[0]=a

        return best