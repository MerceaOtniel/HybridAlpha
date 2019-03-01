from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from tictactoe import TicTacToePlayers as tictacplayers
from othello import OthelloPlayers as othelloplayers
from gobang import GobangPlayers as gobangplayers
from connect4 import Connect4Players as connect4players
import multiprocessing as mp
import copy


class AlphaZeroCoach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args, mcts=True)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.counter = 0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getNoSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    ''' Use this function to decide the players that will be faced by the network at each iteration'''

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        begining=1
        if self.args.load_model == True:

            self.loadTrainExamples()
            file=open(self.args.trainExampleCheckpoint+"loopinformation","r+")
            lines=file.readlines()
            begining=lines[0]
            file.close()


        for i in range(int(begining), self.args.numIters + 1):

            fileLoopInformation = open(self.args.trainExampleCheckpoint + "loopinformation", "w+")
            fileLoopInformation.write(str(i))
            fileLoopInformation.close()

            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()

            for eps in range(self.args.numEps):
                iterationTrainExamples += self.executeEpisode()

                # bookkeeping + plot progress
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                           maxeps=self.args.numEps,
                                                                                                           et=eps_time.avg,
                                                                                                           total=bar.elapsed_td,
                                                                                                           eta=bar.eta_td)
                bar.next()
            bar.finish()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one

            filename = "AlphaZerocurent" + str(i) + "temp:iter" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + \
                       ":dim" + str(self.game.n) + ".pth.tar"

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=filename)

            self.nnet.train(trainExamples)

            self.mcts.clear()
            del self.mcts
            self.mcts = MCTS(self.game, self.nnet, self.args, mcts=True)  # reset search tree

    def getCheckpointFile(self, iteration):
        '''

        :param iteration: the number of current iteration
        :return: a name composed of iteration for saving files
        '''
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):

        '''

        :param iteration: the number of current iteration which will be used to name the example file in an organised
        way
        :return: nothing
        Save the example file in an organised way in order to provide a way of reusing the examples in case of
        unexpected failure
        '''
        folder = self.args.trainExampleCheckpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
            Pickler(f).clear_memo()
        f.close()

    def loadTrainExamples(self):
        '''

        :return: doesn't return anything
        It has the role of loading the examples from the file in order to be reused when the program starts again,
        after maybe a failure, or unexpected interruption
        '''
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.close()