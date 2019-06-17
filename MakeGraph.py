import matplotlib.pyplot as plt



def displayGraphRomanian(wins, draws, lose, information, dimension, name1, name2, name3, name4, title):

    plt.subplot(223)
    plt.title("                "+title)
    plt.plot(wins)  # int_list signifies wins against greedy policy
    plt.plot(draws)
    plt.plot(lose)
    plt.ylabel("numărul de jocuri")
    plt.xlabel("numărul de iteraţii")
    plt.ylim(-0.1, 14.1)
    plt.legend(
        [name1, name2, name3], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.savefig(
        "./temp/tictactoe/graph_rom" + information[1] + ":" + information[2] + ":" + dimension[0] + name4 + ".png")
    plt.show()

def displayGraphEnglish(wins, draws, lose, information, dimension, name1, name2, name3, name4, title):
    plt.subplot(223)
    plt.title("                   "+title)
    plt.plot(wins)  # int_list signifies wins against greedy policy
    plt.plot(draws)
    plt.plot(lose)
    plt.ylabel("number of games")
    plt.xlabel("number of iterations")
    plt.ylim(-0.1, 14.1)
    plt.legend(
        [name1, name2, name3], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(
        "./temp/tictactoe/graph_eng" + information[1] + ":" + information[2] + ":" + dimension[0] + name4 + ".png")
    plt.show()


def prepareGraphForPrint(filename, training=False):


    if training==False:
        information=filename.split("_")
        dimension=information[3].split(".")

        file = open(filename, "r")
        lines=file.readlines()
        for index,line in enumerate(lines):
            v=line.split(" ")
            if index==0:
                int_list=[int(i) for i in v]
            else:
                draws=[int(i) for i in v]
        loses=[]
        for i in range(len(int_list)):
            loses.append(14-int_list[i]-draws[i])

        print(int_list)
        print(draws)

        displayGraphEnglish(int_list, draws, loses, information, dimension,
                     "wins against current network",
                     "draws against current network",
                     "losses against current network",
                     "network",
                     "               Evolution of the best network against the current network" +"\n in each iteration")
        displayGraphRomanian(int_list, draws, loses, information, dimension,
                    "victorii împotriva reţelei curente",
                    "egaluri împotriva reţelei curente",
                    "înfrângeri împotriva reţelei curente",
                    "reţea",
                    "                Evoluţia celei mai bune reţele împotriva reţelei curente " +"\n în fiecare iteraţie")

    else:
        information = filename.split("_")
        dimension = information[3].split(".")

        file = open(filename, "r")

        greedywin=[]
        greedydraw=[]
        greedylose=[]
        randomwin=[]
        randomdraw=[]
        randomlose=[]
        minmaxwin=[]
        minmaxdraw=[]
        minmaxlose=[]

        lines=file.readlines()
        for index,line in enumerate(lines):
            v = line.split(" ")
            if index==0:
                greedywin=[int(i) for i in v]
            elif index==1:
                greedydraw=[int(i) for i in v]
            elif index==2:
                randomwin=[int(i) for i in v]
            elif index==3:
                randomdraw=[int(i) for i in v]
            elif index==4:
                minmaxwin=[int(i) for i in v]
            elif index==5:
                minmaxdraw=[int(i) for i in v]

        for i in range(len(greedywin)):
            greedylose.append(14-greedywin[i]-greedydraw[i])
            randomlose.append(14-randomwin[i]-randomdraw[i])
            minmaxlose.append(14-minmaxwin[i]-minmaxdraw[i])




        displayGraphEnglish(greedywin, greedydraw, greedylose, information, dimension,
                     "number of wins against Greedy",
                     "number of draws against Greedy",
                     "number of losses against Greedy",
                     "greedy",
                     "Evolution of the network against Greedy baseline")
        displayGraphEnglish(randomwin, randomdraw, randomlose, information, dimension,
                     "number of wins against Random",
                     "number of draws against Random",
                     "number of losses against Random",
                     "random",
                     "Evolution of the network against Random baseline")
        displayGraphRomanian(greedywin, greedydraw, greedylose, information, dimension,
                    "numărul de victorii împotriva Greedy",
                    "numărul de egaluri împotriva Greedy",
                    "numărul de înfrângeri împotriva Greedy",
                    "greedy",
                    "Evoluţia reţelei împotriva agentului Greedy")
        displayGraphRomanian(randomwin, randomdraw, randomlose, information, dimension,
                    "numărul de victorii împotriva Random",
                    "numărul de egaluri împotriva Random",
                    "numărul de înfrângeri împotriva Random",
                    "random",
                    "Evoluţia reţelei împotriva agentului Random")

        #afisareGraf(minmaxwin, minmaxdraw, minmaxlose, informatii, dimensiune, "minmaxwin", "minmaxydraw", "minmaxlose","minmax")

prepareGraphForPrint("./temp/tictactoe/graphwins_iter75_eps200_dim8.txt")
prepareGraphForPrint("./temp/tictactoe/graphwins_iter75_eps200_dim8_greedyrandom.txt", True)