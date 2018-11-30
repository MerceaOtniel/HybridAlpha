import numpy as np


class RandomOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)

        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class MinMaxOthelloPlayer():
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

