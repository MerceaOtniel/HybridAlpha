from Coach import Coach

from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as nn

from othello.OthelloGame import OthelloGame as Game1
from othello.keras.NNet import NNetWrapper as nn1

from gobang.GobangGame import GobangGame as Game2
from gobang.keras.NNet import NNetWrapper as nn2

from utils import *


args = dotdict({
    'numIters': 1,
    'numEps': 2,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 10,
    'numMCTSSims': 5,
    'arenaCompare': 5,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})

if __name__=="__main__":

    alegere=2 # chose which game to play 0-tictactoe, 1-othello, 2-gobang

    if alegere==0:
        g = Game()
        nnet = nn(g)
    if alegere==1:
        g = Game1(6)
        nnet = nn1(g)
    if alegere==2:
        g=Game2(6,6)
        nnet = nn2(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
