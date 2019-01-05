import numpy as np
from math import inf as infinity
from tictactoe.TicTacToeGame import Board
import random
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
            #print(str(score)+" "+str(move))
            candidates += [(-score, a)]
        candidates.sort()
        list=[]
        max=candidates[0][0]
        for i in range(len(candidates)):
            if candidates[i][0] == max:
                list.append(candidates[i][1])
        return random.choice(list)


class MinMaxTicTacToePlayer():
    def __init__(self,game):
       self.game=game

    def play(self,board):
        score = self.minimax((board,-1),9,1,-infinity,+infinity)
        list=[]
        for i in range(len(score[0])):
            list.append(score[0][i][0])
        action=random.choice(list)
        return action

    def minimax(self,state,depth,player,alfa,beta):

        best = [[], None]

        if player==1:
            best[1]=-infinity
        else:
            best[1]=+infinity


        if self.game.getGameEnded(state[0],player)!=0:
            score=self.game.getGameEnded(state[0],player)
            return [[],score]
        elif depth==0:
            score=self.game.getScore(state[0],player)
            return [[],score]
        '''
        if depth==0 or self.game.getGameEnded(state[0],player)!=0:
            score=self.game.getGameEnded(state[0],player)
            return [None,score]
        '''

        valids = self.game.getValidMoves(state[0], player)
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard= self.game.getNextState(state[0], player, a)
            score = self.minimax(nextBoard, depth-1, -player,alfa,beta)

            if player==1:
                if score[1] > best[1]:
                    best[1]=score[1]
                    best[0]=[(a,score[1])]
                elif score[1]==best[1]:
                    best[0].append((a,score[1]))
                alfa=max(alfa,best[1])
                if beta<alfa: # here is not equal because i want to select random from multiple actions with the same reward
                    break
            else:
                if score[1]<best[1]:
                    best[1]=score[1]
                    best[0]=[(a,score[1])]
                elif score[1]==best[1]:
                    best[0].append((a,score[1]))
                beta=min(beta,best[1])
                if beta < alfa:
                    break

        return best


