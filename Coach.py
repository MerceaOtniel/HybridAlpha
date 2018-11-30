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

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

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
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]



    ''' Use this function to decide the players that will be faced by the network at each iteration'''
    def decidePlayers(self):

        if "tictactoe" in self.args.trainExampleCheckpoint:
            rp = tictacplayers.RandomTicTacToePlayer(self.game).play
            gp = tictacplayers.GreedyTicTacToePlayer(self.game).play
            mp = tictacplayers.MinMaxTicTacToePlayer(self.game).play
        else:
            if "othello" in self.args.trainExampleCheckpoint:
                gp = othelloplayers.GreedyOthelloPlayer(self.game).play
                rp = othelloplayers.RandomOthelloPlayer(self.game).play
                mp = othelloplayers.MinMaxOthelloPlayer(self.game).play
            else:
                if "gobang" in self.args.trainExampleCheckpoint:
                    gp = gobangplayers.GreedyGobangPlayer(self.game).play
                    rp = gobangplayers.RandomGobangPlayer(self.game).play
                    mp = gobangplayers.MinMaxGobangPlayer(self.game).play
                else:
                    if "connect4" in self.args.trainExampleCheckpoint:
                        rp = connect4players.RandomConnect4Player(self.game).play
                        gp = connect4players.GreedyConnect4Player(self.game).play
                        mp = tictacplayers.GreedyTicTacToePlayer(self.game).play

        return (gp, rp, mp)


    ''' Use this function to write to file the number of draws/wins '''
    def writeToFile(self,file,epochs):
        for text in epochs:
            file.write(str(text)+" ")
        file.write("\n")


    ''' 
    True means that the training is written to file; false means that the pit is written to file
    Pass this function the output of the agents and it should write it into a file
    '''
    def writeLogsToFile(self,epochswin,epochdraw,epochswin2=[],epochsdraw2=[],epochswin3=[],epochsdraw3=[],training=True):
        if training==True:
            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ".txt", "w+")
            print("Path-ul este " + str(file))
            for text in epochswin:
                file.write(str(text) + " ")
            file.write("\n")
            for text in epochdraw:
                file.write(str(text) + " ")
            file.close()
        else:
            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ":greedyrandom.txt", "w+")
            print("Path-ul este " + str(file))

            self.writeToFile(file,epochswin)
            self.writeToFile(file, epochdraw)
            self.writeToFile(file, epochswin2)
            self.writeToFile(file, epochsdraw2)
            self.writeToFile(file, epochswin3)
            self.writeToFile(file, epochsdraw3)
            file.close()


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        epochswin=[] # count the number of wins at every epoch of the network against the preceding version
        epochdraw=[] # count the number of draws at every epoch of the network against the preceding version
        epochswingreedy=[] # count the number of wins against greedy at every epoch
        epochswinrandom=[] # count the number of wins against random at every epoch
        epochsdrawgreedy=[] #count the number of draws against greedy at every epoch
        epochsdrawrandom=[]  #count the number of wins against random at every epoch
        epochswinminmax=[] #count the number of wins against minmax at every epoch
        epochsdrawminmax=[] #count the number of draws against minmax at every epoch

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    iterationTrainExamples += self.executeEpisode()
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one

            filename = "temp:iter" + str(self.args.numIters) + ":eps"+str(self.args.numEps) + ":dim" + str(self.game.n) + ".pth.tar"
            filenameBest = "best" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(self.game.n) + ".pth.tar"

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=filename)
            exists = os.path.isfile(filenameBest)
            if exists:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=filenameBest)
            else:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=filename)
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            print(' ')
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if i==1:
                epochswin.append(pwins)
                epochdraw.append(0)

            epochswin.append(nwins)
            epochdraw.append(draws)
            self.writeLogsToFile(epochswin,epochdraw)


            ''' Get all the players and then put them against the network'''
            (gp, rp, mp) = self.decidePlayers()

            arenagreedy = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), gp, self.game)
            arenarandom = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), rp, self.game)
            arenaminmax = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), mp, self.game)

            pwinsminmax,nwinsminmax,drawsminmax=arenaminmax.playGames(self.args.arenaCompare)
            pwinsgreedy,nwinsgreedy,drawsgreedy=arenagreedy.playGames(self.args.arenaCompare)
            pwinsreandom,nwinsrandom,drawsrandom=arenarandom.playGames(self.args.arenaCompare)

            epochsdrawgreedy.append(drawsgreedy)
            epochsdrawrandom.append(drawsrandom)
            epochswinrandom.append(pwinsreandom)
            epochswingreedy.append(pwinsgreedy)
            epochswinminmax.append(pwinsminmax)
            epochsdrawminmax.append(drawsminmax)


            self.writeLogsToFile(epochswingreedy,epochsdrawgreedy,epochswinrandom,epochsdrawrandom,epochswinminmax,epochsdrawminmax,training=False)


            if pwins+nwins == 0 or float(nwins)/(pwins+nwins+draws) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                filename = "temp:iter" + str(self.args.numIters) +":eps"+str(self.args.numEps) + ":dim"+str(self.game.n) + ".pth.tar"
                filenameBest = "best" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(self.game.n) + ".pth.tar"
                exists = os.path.isfile(filenameBest)
                if exists:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=filenameBest)
                else:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=filename)

            else:
                print('ACCEPTING NEW MODEL')
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                filename="best"+ str(self.args.numIters) +":eps"+str(self.args.numEps) + ":dim"+str(self.game.n) +".pth.tar"
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=filename)

        self.writeLogsToFile(epochswin,epochdraw,training=True)


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.trainExampleCheckpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
