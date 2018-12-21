import matplotlib.pyplot as plt



def afisareGraf(wins,draws,lose,informatii,dimensiune,name1,name2,name3,name4):

    plt.title("Evolutia retelei in comparatie cu celelalte policies " + informatii[1] + " " + informatii[2] + " " +
              dimensiune[0])
    plt.subplot(223)
    plt.plot(wins)  # int_list signifies wins against greedy policy
    plt.plot(draws)
    plt.plot(lose)
    plt.ylabel("numar jocuri")
    plt.xlabel("numar epoci")
    plt.ylim(-1, 41)
    plt.legend(
        [name1, name2, name3], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(
        "./temp/tictactoe/graph" + informatii[1] + ":" + informatii[2] + ":" + dimensiune[0] + name4+".png")
    plt.show()



def paintGraph(filename,training=False):


    if training==False:
        informatii=filename.split(":")
        dimensiune=informatii[3].split(".")

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
            loses.append(40-int_list[i]-draws[i])

        print(int_list)
        print(draws)

        afisareGraf(int_list,loses,draws,informatii,dimensiune,"castigate","pierdute","remize","retea")

    else:
        informatii = filename.split(":")
        dimensiune = informatii[3].split(".")

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
            greedylose.append(40-greedywin[i]-greedydraw[i])
            randomlose.append(40-randomwin[i]-randomdraw[i])
            minmaxlose.append(40-minmaxwin[i]-minmaxdraw[i])


        plt.title("Evolutia retelei in comparatie cu celelalte policies "+informatii[1]+" "+informatii[2]+" "+dimensiune[0])

        afisareGraf(greedywin,greedydraw,greedylose,informatii,dimensiune,"greedywin","greedydraw","greedylose","greedy")
        afisareGraf(randomwin, randomdraw, randomlose, informatii, dimensiune, "randomwin", "randomdraw", "randomlose","random")
        afisareGraf(minmaxwin, minmaxdraw, minmaxlose, informatii, dimensiune, "minmaxwin", "minmaxydraw", "minmaxlose","minmax")


paintGraph("./temp/tictactoe/graphwins:iter75:eps50:dim3.txt")
paintGraph("./temp/tictactoe/graphwins:iter75:eps50:dim3:greedyrandom.txt",True)
