import matplotlib.pyplot as plt


def paintGraph(filename,training=False):


    if training==False:
        informatii=filename.split(":")
        dimensiune=informatii[3].split(".")

        file = open(filename, "r")
        i=0

        for line in file.readlines():
            v=line.split(" ")
            if i==0:
                int_list = [int(i) for i in v]
            else:
                draws=[int(i) for i in v]
            i+=1
        loses=[]
        for i in range(len(int_list)):
            loses.append(40-int_list[i]-draws[i])

        print(int_list)
        print(draws)

        plt.title("Evolutia retelei in timpul antrenamentului "+informatii[1]+" "+informatii[2]+" "+dimensiune[0])
        plt.subplot(223)
        plt.plot(int_list)
        plt.plot(loses)
        plt.plot(draws)
        plt.ylabel('numar jocuri')
        plt.xlabel('numar epoci')
        plt.ylim(-1,41)
        plt.legend(['castigate', 'pierdute', 'remize'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("./temp/tictactoe/graph"+informatii[1]+":"+informatii[2]+":"+dimensiune[0]+".png")
        plt.show()
    else:
        informatii = filename.split(":")
        dimensiune = informatii[3].split(".")

        file = open(filename, "r")
        i = 0

        greedywin=[]
        greedydraw=[]
        greedylose=[]
        randomwin=[]
        randomdraw=[]
        randomlose=[]

        for line in file.readlines():
            v = line.split(" ")
            if i==0:
                greedywin=[int(i) for i in v]
            else:
                if i==1:
                    greedydraw=[int(i) for i in v]
                else:
                    if i==2:
                        randomwin=[int(i) for i in v]
                    else:
                        if i==3:
                            randomdraw=[int(i) for i in v]
            i+=1
        for i in range(len(greedywin)):
            greedylose.append(40-greedywin[i]-greedydraw[i])
            randomlose.append(40-randomwin[i]-randomdraw[i])


        plt.title("Evolutia retelei in comparatie cu celelalte policies "+informatii[1]+" "+informatii[2]+" "+dimensiune[0])
        plt.subplot(223)
        plt.plot(greedywin) #int_list signifies wins against greedy policy
        plt.plot(greedydraw)
        plt.plot(greedylose)
        plt.plot(randomwin) # draws signifies wins against random policy
        plt.plot(randomdraw)
        plt.plot(randomlose)
        plt.ylabel("numar jocuri")
        plt.xlabel("numar epoci")
        plt.ylim(-1,41)
        plt.legend(['greedywin','greedydraw','greedylose','randomwin','randomdraw','randomlose'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("./temp/tictactoe/graph"+informatii[1]+":"+informatii[2]+":"+dimensiune[0]+"greedyrandom.png")
        plt.show()


paintGraph("./temp/tictactoe/graphwins:iter5:eps2:dim3.txt")
paintGraph("./temp/tictactoe/graphwins:iter5:eps2:dim3:greedyrandom.txt",True)
