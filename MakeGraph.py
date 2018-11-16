import matplotlib.pyplot as plt

file = open("./temp/tictactoe/graphwins:iter25:eps27:dim4.txt", "r")
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

plt.title("Evolutia retelei in timpul antrenamentului")
plt.plot(int_list)
plt.plot(loses)
plt.plot(draws)
plt.ylabel('numar jocuri')
plt.xlabel('numar epoci')
plt.ylim(-1,41)
plt.legend(['jocuri castigate', 'jocuri pierdute', 'numar remize'], loc='right')
plt.show()
