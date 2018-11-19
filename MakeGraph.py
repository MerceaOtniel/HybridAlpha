import matplotlib.pyplot as plt


def paintGraph(filename,training=False):

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

    if training==False:
        loses=[]
        for i in range(len(int_list)):
            loses.append(40-int_list[i]-draws[i])

        print(int_list)
        print(draws)

        plt.title("Evolutia retelei in timpul antrenamentului "+informatii[1]+" "+informatii[2]+" "+dimensiune[0])
        plt.plot(int_list)
        plt.plot(loses)
        plt.plot(draws)
        plt.ylabel('numar jocuri')
        plt.xlabel('numar epoci')
        plt.ylim(-1,41)
        plt.legend(['jocuri castigate', 'jocuri pierdute', 'numar remize'], loc='right')
        plt.savefig("./temp/tictactoe/graph"+informatii[1]+":"+informatii[2]+":"+dimensiune[0]+".png")
        plt.show()
    else:
        plt.title("Evolutia retelei in comparatie cu celelalte policies "+informatii[1]+" "+informatii[2]+" "+dimensiune[0])
        plt.plot(int_list) #int_list signifies wins against greedy policy
        plt.plot(draws) # draws signifies wins against random policy
        plt.ylabel("numar jocuri")
        plt.xlabel("numar epoci")
        plt.ylim(-1,41)
        plt.legend(['castigate impotriva greedy', 'castigate impotriva random'], loc='right')
        plt.savefig("./temp/tictactoe/graph"+informatii[1]+":"+informatii[2]+":"+dimensiune[0]+"greedyrandom.png")
        plt.show()


paintGraph("./temp/tictactoe/graphwins:iter25:eps2:dim4.txt")
paintGraph("./temp/tictactoe/graphwins:iter25:eps2:dim4:greedyrandom.txt",True)
