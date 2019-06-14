from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .OthelloLogic import Board
import numpy as np


class OthelloGame(Game):
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1*player
        if b.countDiff(player) == 0:
            return 1e-4
        return -1*player

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def getNoSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        l = [(board, pi)]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()


    def moveNumberHeuristics(self,color,board):
        b = Board(self.n)
        b1=Board(self.n)
        b.pieces = np.copy(board)
        b1.pieces=np.copy(board)
        legalMoves = b.get_legal_moves(color)
        legalMoves1=b1.get_legal_moves(-color)

        numberMovesPlayer=len(legalMoves)
        numberMovesOpponent=len(legalMoves1)

        if color==-1:
            aux=numberMovesOpponent
            numberMovesOpponent=numberMovesPlayer
            numberMovesPlayer=aux

        if numberMovesPlayer+numberMovesOpponent==0:
            return 0
        return (numberMovesPlayer-numberMovesOpponent)/(numberMovesPlayer+numberMovesOpponent)

    def cornerNumberHeuristics(self,color,board):

        playerCorners=0
        adversaryCorners=0

        if board[0][0]==color:
            playerCorners+=1
        else:
            if board[0][0]==-color:
                adversaryCorners+=1

        if board[0][self.n-1]==color:
            playerCorners+=1
        else:
            if board[0][self.n-1]==-color:
                adversaryCorners+=1

        if board[self.n-1][0]==color:
            playerCorners+=1
        else:
            if board[self.n-1][0]==-color:
                adversaryCorners+=1

        if board[self.n-1][self.n-1]==color:
            playerCorners+=1
        else:
            if board[self.n-1][self.n-1]==-color:
                adversaryCorners+=1

        if color==-1:
            aux=adversaryCorners
            adversaryCorners=playerCorners
            playerCorners=aux

        if playerCorners+adversaryCorners==0:
            return 0
        return (playerCorners-adversaryCorners)/(playerCorners+adversaryCorners)

    def countPiecesHeuristics(self, color,board):
        count1 = 0
        count2 = 0
        for y in range(self.n):
            for x in range(self.n):
                if board[x][y] == color:
                    count1 += 1
                if board[x][y] == -color:
                    count2 += 1

        if color==-1:
            aux=count1
            count1=count2
            count2=aux

        if count1+count2==0:
            return 0
        return (count1 - count2) / (count1 + count2)

    def getScore(self, board, player):
        countPieces=self.countPiecesHeuristics(player,board)
        countCorners=self.cornerNumberHeuristics(player,board)
        countMoves=self.moveNumberHeuristics(player,board)
        return (countMoves+countPieces+countCorners)/3

def display(board):
    n = board.shape[0]

    print("   ", end="")
    for y in range(n):
        print(y, "", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|", end="")  # print the row #
        for x in range(n):
            piece = board[y][x]  # get the piece to print
            if piece == -1:
                print("B ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
