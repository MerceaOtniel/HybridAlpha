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
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    ''' Use this function to decide the players that will be faced by the network at each iteration'''

    def decidePlayers(self):

        '''

        :return: the benchmarks(agents) for the desired game
        '''

        if "tictactoe" in self.args.trainExampleCheckpoint:
            rp = tictacplayers.RandomTicTacToePlayer(self.game).play
            gp = tictacplayers.GreedyTicTacToePlayer(self.game).play
            mp = tictacplayers.MinMaxTicTacToePlayer(self.game,4).play #it creates by default a minmax with depth4
        else:
            if "othello" in self.args.trainExampleCheckpoint:
                gp = othelloplayers.GreedyOthelloPlayer(self.game).play
                rp = othelloplayers.RandomOthelloPlayer(self.game).play
                mp = othelloplayers.MinMaxOthelloPlayer(self.game,4).play
            else:
                if "gobang" in self.args.trainExampleCheckpoint:
                    gp = gobangplayers.GreedyGobangPlayer(self.game).play
                    rp = gobangplayers.RandomGobangPlayer(self.game).play
                    mp = gobangplayers.MinMaxGobangPlayer(self.game,3).play
                else:
                    if "connect4" in self.args.trainExampleCheckpoint:
                        rp = connect4players.RandomConnect4Player(self.game).play
                        gp = connect4players.GreedyConnect4Player(self.game).play
                        mp = connect4players.MinMaxConnect4Player(self.game,3).play

        return (gp, rp, mp)

    ''' Use this function to write to file the number of draws/wins '''

    def writeToFile(self, file, epochs):

        '''

        :param file: the file where data will be written
        :param epochs: the list of values that will be written in the file
        :return: nothing
        '''

        for text in epochs:
            file.write(str(text) + " ")
        file.write("\n")

    ''' 
    True means that the training is written to file; false means that the pit is written to file
    Pass this function the output of the agents and it should write it into a file
    '''

    def writeLogsToFile(self, epochswin, epochdraw, epochswin2=[], epochsdraw2=[], epochswin3=[], epochsdraw3=[],
                        training=True):


        '''

        :param epochswin: network wins against greedy agent/ or against itself if training=True
        :param epochdraw: network draws against greedy agent/ or against itself if training=True
        :param epochswin2: network wins against random agent
        :param epochsdraw2: network draws against random agent
        :param epochswin3: network wins against minimax agent
        :param epochsdraw3: network draws against minimax agent
        :param training: specifies if we evaluate the network against itself, or against the benchmakrs mentioned
        above
        :return: nothing
        This function has the role of logging the progress of the network to 2 files in order to plot it
        '''


        if training == True:
            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ".txt", "w+")
            print("Path-ul este " + str(file))
            self.writeToFile(file, epochswin)
            self.writeToFile(file, epochdraw)
            file.close()
        else:
            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ":greedyrandom.txt", "w+")
            print("Path-ul este " + str(file))

            self.writeToFile(file, epochswin)
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
        epochswin = []  # count the number of wins at every epoch of the network against the preceding version
        epochdraw = []  # count the number of draws at every epoch of the network against the preceding version
        epochswingreedy = []  # count the number of wins against greedy at every epoch
        epochswinrandom = []  # count the number of wins against random at every epoch
        epochsdrawgreedy = []  # count the number of draws against greedy at every epoch
        epochsdrawrandom = []  # count the number of wins against random at every epoch
        epochswinminmax = []  # count the number of wins against minmax at every epoch
        epochsdrawminmax = []  # count the number of draws against minmax at every epoch

        begining=1
        if self.args.load_model == True:
            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ".txt", "r+")
            lines = file.readlines()
            for index, line in enumerate(lines):
                for word in line.split():
                    if index == 0:
                        epochswin.append(word)
                    elif index == 1:
                        epochdraw.append(word)
            file.close()

            file = open(self.args.trainExampleCheckpoint + "graphwins:iter" + str(self.args.numIters) + ":eps" + str(
                self.args.numEps) + ":dim" + str(self.game.n) + ":greedyrandom.txt", "r+")
            lines = file.readlines()
            for index, line in enumerate(lines):
                for word in line.split():
                    if index == 0:
                        epochswingreedy.append(word)
                    elif index == 1:
                        epochsdrawgreedy.append(word)
                    elif index == 2:
                        epochswinrandom.append(word)
                    elif index == 3:
                        epochsdrawrandom.append(word)
                    elif index == 4:
                        epochswinminmax.append(word)
                    elif index == 5:
                        epochsdrawminmax.append(word)
            file.close()
            self.loadTrainExamples()

            file=open(self.args.trainExampleCheckpoint+"loopinformation","r+")
            lines=file.readlines()
            begining=lines[0]
            file.close()


        for i in range(int(begining), self.args.numIters + 1):
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

            fileLoopInformation = open(self.args.trainExampleCheckpoint+"loopinformation","w+")
            fileLoopInformation.write(str(i))
            fileLoopInformation.close()

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

            filename = "curent"+str(i)+"temp:iter" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + \
                       ":dim" + str(self.game.n) + ".pth.tar"
            filenameBest = "best" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(
                self.game.n) + ".pth.tar"
            print("path with filename "+filename)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=filename)
            exists = os.path.isfile(filenameBest)
            if exists:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=filenameBest)
            else:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=filename)
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            filenameCurrent="currentforprocess:temp:iter" + str(self.args.numIters) + \
                            ":eps" + str(self.args.numEps) + ":dim" + str(self.game.n) + ".pth.tar"
            self.nnet.save_checkpoint(folder=self.args.checkpoint,filename=filenameCurrent)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game,nmcts,pmcts,evaluate=True,
                          name=self.args.name)

            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, False)

            pmcts.clear()
            nmcts.clear()
            del pmcts
            del nmcts

            print(' ')
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if i == 1:
                epochswin.append(pwins)
                epochdraw.append(0)

            epochswin.append(nwins)
            epochdraw.append(draws)
            self.writeLogsToFile(epochswin, epochdraw)

            ''' Get all the players and then pit them against the network. You need to modify here if you implement 
                more players
            '''
            (gp, rp, mp) = self.decidePlayers()

            if self.args.parallel == 0:


                nmcts1 = MCTS(self.game, self.nnet, self.args)
                nmcts2 = MCTS(self.game, self.nnet, self.args)
                nmcts3 = MCTS(self.game, self.nnet, self.args)

                arenagreedy = Arena(lambda x: np.argmax(nmcts1.getActionProb(x, temp=0)), gp, self.game,nmcts1
                                    ,name=self.args.name)
                arenarandom = Arena(lambda x: np.argmax(nmcts2.getActionProb(x, temp=0)), rp, self.game,nmcts2
                                    ,name=self.args.name)
                arenaminmax = Arena(lambda x: np.argmax(nmcts3.getActionProb(x, temp=0)), mp, self.game,nmcts3,
                                    evaluate=True,name=self.args.name)

                pwinsminmax, nwinsminmax, drawsminmax = arenaminmax.playGames(self.args.arenaCompare)
                print("minmax - "+str(pwinsminmax)+" "+str(nwinsminmax)+" "+str(drawsminmax))
                pwinsgreedy, nwinsgreedy, drawsgreedy = arenagreedy.playGames(self.args.arenaCompare)
                print("greedy - "+str(pwinsgreedy)+" "+str(nwinsgreedy)+" "+str(drawsgreedy))
                pwinsreandom, nwinsrandom, drawsrandom = arenarandom.playGames(self.args.arenaCompare)
                print("random - "+str(pwinsreandom)+" "+str(nwinsrandom)+" "+str(drawsrandom))

                nmcts1.clear()
                nmcts2.clear()
                nmcts3.clear()
                del nmcts1
                del nmcts2
                del nmcts3

            else:
                '''
                This will be used if you want to evaluate the network against the benchmarks in a parallel way
                '''

                self.args.update({'index': str(i)})

                p = self.parallel(self.args.arenaCompare)
                (pwinsminmax, nwinsminmax, drawsminmax) = p[0]  # self.parallel("minmax", self.args.arenaCompare)
                (pwinsgreedy, nwinsgreedy, drawsgreedy) = p[1]  # self.parallel("greedy",self.args.arenaCompare)
                (pwinsreandom, nwinsrandom, drawsrandom) = p[2]  # self.parallel("random",self.args.arenaCompare)

            epochsdrawgreedy.append(drawsgreedy)
            epochsdrawrandom.append(drawsrandom)
            epochswinrandom.append(pwinsreandom)
            epochswingreedy.append(pwinsgreedy)
            epochswinminmax.append(pwinsminmax)
            epochsdrawminmax.append(drawsminmax)

            self.writeLogsToFile(epochswingreedy, epochsdrawgreedy, epochswinrandom, epochsdrawrandom, epochswinminmax,
                                 epochsdrawminmax, training=False)

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) <= self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                filename = "curent"+str(i)+"temp:iter" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(
                    self.game.n) + ".pth.tar"
                filenameBest = "best" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(
                    self.game.n) + ".pth.tar"
                exists = os.path.isfile(filenameBest)
                if exists:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=filenameBest)
                else:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=filename)

            else:
                print('ACCEPTING NEW MODEL')
                filename = "best" + str(self.args.numIters) + ":eps" + str(self.args.numEps) + ":dim" + str(
                    self.game.n) + ".pth.tar"
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=filename)
            self.mcts.clear()
            del self.mcts
            self.mcts = MCTS(self.game, self.nnet, self.args, mcts=True)  # reset search tree
        self.writeLogsToFile(epochswin, epochdraw, training=True)


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

    def parallel(self, num):

        '''
        :param num: the number of iterations for each agent
        :return: the score for random agent, minimax agent, greedy agent

        This function creates 4 processes: 2 processes for minimax that each will analyze half of games
                                           1 process for greedy
                                           1 process for random
        In this way the whole evaluation is cut with somewhere around 40%, but it uses one gpu, so be aware if it goes
        hot.

        '''


        first_half = num / 2
        second_half = num / 2 + num % 2

        args = copy.deepcopy(self.args)

        if self.counter == 0:
            mp.set_start_method('spawn')

        qminmax = mp.Queue()
        qrandom = mp.Queue()
        qgreedy = mp.Queue()
        qminmax1 = mp.Queue()

        self.counter += 1

        processminmax = startprocess(callminmax, first_half, qminmax, args)
        processminmax.join()
        processminmax1 = startprocess(callminmax, second_half, qminmax1, args)
        processminmax1.join()
        processrandom = startprocess(callrandom, num, qrandom, args)
        processrandom.join()
        processgreedy = startprocess(callgreedy, num, qgreedy, args)
        processgreedy.join()


        (pwinsminmax1, nwinsminmax1, drawsminmax1) = verifyqueue(callminmax, first_half, qminmax, args)
        (pwinsminmax2, nwinsminmax2, drawsminmax2) = verifyqueue(callminmax, second_half, qminmax1, args)
        pwinsminmax = pwinsminmax2 + pwinsminmax1
        nwinsminmax = nwinsminmax2 + nwinsminmax1
        drawsminmax = drawsminmax2 + drawsminmax1
        (pwinsrandom, nwinsrandom, drawsrandom) = verifyqueue(callrandom, num, qrandom, args)
        (pwinsgreedy, nwinsgreedy, drawsgreedy) = verifyqueue(callgreedy, num, qgreedy, args)

        u = [(pwinsminmax, nwinsminmax, drawsminmax), (pwinsgreedy, nwinsgreedy, drawsgreedy),
             (pwinsrandom, nwinsrandom, drawsrandom)]

        return u


def startprocess(function, num, q, args):
    '''

    :param function: which agent to use
    :param num: how many games should that agent play
    :param q: the queue where the number of wins, loses and draws are stored
    :param args: config for games and neural network,etc
    :return: a variable which you can control the process

    This function start the process and gives all the necessarily details
    '''
    p = mp.Process(target=function, args=(num, q, args,))
    p.start()
    return p


def verifyqueue(function, num, q, args):

    '''

    :param function: the agent which will be used
    :param num: the number of games
    :param q: the queue that will store the results
    :param args: configs
    :return: the number of wins/loses/draws

    THis function has the responsibility of checking if the process played some games, sometimes
    tensorflow will not be able to create a process and if this happens, q will be empty.
    It tries to start a process while q is empty. When q becomes non-empty, it means that a process containing tensorflow
    has been created

    '''

    while q.empty() == True:
        p = startprocess(function, num, q, args)
        p.join()
    return verifyvalues(function, num, q, args)


def verifyvalues(function, num, q, args):

    '''
    :param function: the agent which will be used
    :param num: the number of games
    :param q: the queue that will store the results
    :param args: configs
    :return: the number of wins/loses/draws
    Sometimes when it cannot create a new process for tensorflow the queue will not be empty, but draws wins and loses
    will all be 0 which is impossible. So it checks for this and tries to recreate the process again if it fails.
    '''

    (pwins, nwins, draws) = extractvaluefromqueue(q)
    while pwins == 0 and nwins == 0 and draws == 0:
        p = startprocess(function, num, q, args)
        p.join()
        (pwins, nwins, draws) = extractvaluefromqueue(q)
    return (pwins, nwins, draws)


def extractvaluefromqueue(q):

    '''

    :param q: the queue that will store the results
    :return: the results
    '''

    (pwins1, nwins1, draws1) = q.get()
    while q.empty() == False:
        (pwins1, nwins1, draws1) = q.get()
    return (pwins1, nwins1, draws1)


def callminmax(num, q, args):

    '''

    :param num: number of games
    :param q: the queue that will store the results
    :param args: configs
    :return: doesn't return anything, the results are stored in q

    It uses the minmax agent to play the specified games
    '''

    from tictactoe.TicTacToeGame import TicTacToeGame as Game
    from tictactoe.tensorflow.NNet import NNetWrapper as nn
    verify = 0
    while verify == 0:
        try:
            g = Game(3)
            nnet = nn(g, 0.06)

            filenameCurrent = "currentforprocess:temp:iter" + str(args.numIters) + \
                              ":eps" + str(args.numEps) + ":dim" + str(g.n) + ".pth.tar"

            nnet.load_checkpoint(folder=args.checkpoint, filename=filenameCurrent)

            mp = returnplayer(args, "minmax", g)
            nmcts1 = MCTS(g, nnet, args)
            arenaminmax = Arena(lambda x: np.argmax(nmcts1.getActionProb(x, temp=0)), mp, g,nmcts1,evaluate=True)
            pwins, nwins, drawwins = arenaminmax.playGames(num)
            q.put((pwins, nwins, drawwins))
            nmcts1.clear()
            verify = 1
        except:
            verify = 0


def callrandom(num, q, args):

    '''

    :param num: number of games
    :param q: the queue that will store the results
    :param args: configs
    :return: doesn't return anything, the results are stored in q

    It uses the random agent to play the specified games
    '''

    from tictactoe.TicTacToeGame import TicTacToeGame as Game
    from tictactoe.tensorflow.NNet import NNetWrapper as nn
    verify = 0
    while verify == 0:
        try:
            g = Game(3)
            nnet = nn(g, 0.06)

            filenameCurrent = "currentforprocess:temp:iter" + str(args.numIters) + \
                              ":eps" + str(args.numEps) + ":dim" + str(g.n) + ".pth.tar"

            nnet.load_checkpoint(folder=args.checkpoint, filename=filenameCurrent)

            rp = returnplayer(args, "random", g)
            nmcts1 = MCTS(g, nnet, args)
            arenarandom = Arena(lambda x: np.argmax(nmcts1.getActionProb(x, temp=0)), rp, g)
            pwins, nwins, drawwins = arenarandom.playGames(num)
            q.put((pwins, nwins, drawwins))
            nmcts1.clear()
            verify = 1
        except:
            verify = 0


def callgreedy(num, q, args):
    '''

    :param num: number of games
    :param q: the queue that will store the results
    :param args: configs
    :return: doesn't return anything, the results are stored in q

    It uses the greedy agent to play the specified games
    '''

    from tictactoe.TicTacToeGame import TicTacToeGame as Game
    from tictactoe.tensorflow.NNet import NNetWrapper as nn
    verify = 0
    while verify == 0:
        try:
            g = Game(3)
            nnet = nn(g, 0.06)
            filenameCurrent = "currentforprocess:temp:iter" + str(args.numIters) + \
                              ":eps" + str(args.numEps) + ":dim" + str(g.n) + ".pth.tar"

            nnet.load_checkpoint(folder=args.checkpoint, filename=filenameCurrent)

            gp = returnplayer(args, "greedy", g)
            nmcts1 = MCTS(g, nnet, args)
            arenagreedy = Arena(lambda x: np.argmax(nmcts1.getActionProb(x, temp=0)), gp, g)
            pwins, nwins, drawwins = arenagreedy.playGames(num)
            q.put((pwins, nwins, drawwins))
            nmcts1.clear()
            verify = 1
        except:
            verify = 0


def returnplayer(args, playertype, g):

    '''

    Whenever adding new players and games this method needs to be updated
    :param args: configs
    :param playertype: which agent to use
    :param g: game, and is needed to initialize an agent
    :return: the specified agent
    '''

    if args.name == "tictactoe":
        if playertype == "greedy":
            return tictacplayers.GreedyTicTacToePlayer(g).play
        elif playertype == "random":
            return tictacplayers.RandomTicTacToePlayer(g).play
        elif playertype == "minmax":
            return tictacplayers.MinMaxTicTacToePlayer(g,4).play

    elif args.name == "othello":

        if playertype == "greedy":
            return othelloplayers.GreedyOthelloPlayer(g).play
        elif playertype == "random":
            return othelloplayers.RandomOthelloPlayer(g).play
        elif playertype == "minmax":
            return othelloplayers.MinMaxOthelloPlayer(g,4).play

    elif args.name == "gobang":
        if playertype == "greedy":
            return gobangplayers.GreedyGobangPlayer(g).play
        elif playertype == "random":
            return gobangplayers.RandomGobangPlayer(g).play
        elif playertype == "minmax":
            return gobangplayers.MinMaxGobangPlayer(g,4).play

    elif args.name == "connect4":
        if playertype == "greedy":
            return connect4players.GreedyConnect4Player(g).play
        elif playertype == "random":
            return connect4players.RandomConnect4Player(g).play
        elif playertype == "minmax":
            return connect4players.MinMaxConnect4Player(g,4).play