'''
Author: MBoss
Date: Jan 17, 2018.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
class Board():
    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, n,n_in_row):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.moveNumber=0
        self.n_in_row=n_in_row
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n


    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        if self.moveNumber==0:
            for i in range(self.n):
                for j in range(self.n):
                    if self[i][j]!=0:
                        self.moveNumber+=1

        if self.moveNumber==0:
            moves.add((int(self.n/2),int(self.n/2)))
            return list(moves)
        # for a moment it is scripted  to +-1
        if self.moveNumber==2:
            centerX=int(self.n/2)
            centerY=int(self.n/2)
            minX=centerX-1
            maxX=centerX+1
            minY=centerY-1
            maxY=centerY+1
            for y in range(self.n):
                for x in range(self.n):
                    if self[x][y]==0 and (x<minX or x>maxX) and (y<minY or y>maxY):
                        moves.add((x,y))
            return list(moves)


        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.add((x, y))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def execute_move(self, move, color):


         self[move[0]][move[1]]=color


    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod

    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

    def check_number_moves(self,x,y,player):

        max=0
        for i in range(self.n):
            for j in range(self.n):
                numbercontiguous = 0
                pozx=i
                pozy=j
                for u in range(self.n_in_row):
                    if not(pozx>=0 and pozx<self.n):
                        break
                    if not(pozy>=0 and pozy<self.n):
                        break
                    if self[pozx][pozy]==player:
                        numbercontiguous+=1
                    elif self[pozx][pozy]==-player:
                        numbercontiguous=0
                        break
                    else:
                        pozprovx=pozx+x
                        pozprovy=pozy+y
                        if pozprovx<self.n and pozprovy<self.n and self[pozprovx][pozprovy]!=player:
                            break
                    pozx+=x
                    pozy+=y

                if numbercontiguous>max:
                    max=numbercontiguous
        return max


    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""


        maxadv=0
        maxadv=max(maxadv,self.check_number_moves(1,0,-color))
        maxadv=max(maxadv,self.check_number_moves(0,1,-color))
        maxadv=max(maxadv,self.check_number_moves(1,1,-color))
        maxadv=max(maxadv,self.check_number_moves(1,-1,-color))

        if maxadv ==self.n_in_row:
            return -color

        maxplayer=0

        maxplayer = max(maxplayer, self.check_number_moves(1, 0, color))
        maxplayer = max(maxplayer, self.check_number_moves(0, 1, color))
        maxplayer = max(maxplayer, self.check_number_moves(1, 1, color))
        maxplayer = max(maxplayer, self.check_number_moves(1, -1, color))

        if maxplayer==self.n_in_row:
            return color

        if maxplayer+maxadv==0:
            return 0
        return ((maxplayer-maxadv)/(maxplayer+maxadv))*color
