import numpy as np
import random
import os
class TTT:
    def __init__(self):
        self.puzzleHistory = np.zeros((1,9))
        self.history = []
        self.BaseArray = np.zeros((3,3))
        self.x_rand = random.randint(0,2)
        self.y_rand = random.randint(0,2)
        self.BaseArray[self.x_rand][self.y_rand] = -1
        z = [self.x_rand,self.y_rand]
        self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.BaseArray,(1,9)),axis=0) 
        
        self.history.append(z)
        print(self.BaseArray)

    def moveRec(self):
        return self.history
    def puzzleRec(self):
        return self.puzzleHistory
    def move(self,moves):
        self.ComputeArray = self.BaseArray

        x,y = moves
        
        self.ComputeArray[x][y] = 1
        
        self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)        
        self.history.append([x,y])

    def predict(self):
        vertiSums = np.sum(self.ComputeArray, axis=0)
        horiSums = np.sum(self.ComputeArray, axis=1) 
        if(self.ComputeArray[0][0]+self.ComputeArray[1][1]+self.ComputeArray[2][2]==3):
            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)        
    
            return self.ComputeArray,2
        if(self.ComputeArray[0][2]+self.ComputeArray[1][1]+self.ComputeArray[2][0]==3):
            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
            return self.ComputeArray,2
        if((self.ComputeArray[0][0]==-1 and self.ComputeArray[1][1]==-1 and self.ComputeArray[2][2]==0) or ( self.ComputeArray[0][0]==-1 and self.ComputeArray[2][2] ==-1 and self.ComputeArray[1][1]==0) or ( self.ComputeArray[1][1]==-1 and self.ComputeArray[2][2] ==-1 and self.ComputeArray[0][0]==0)):
            
            if self.ComputeArray[0][0]==0 and self.ComputeArray[1][1]==0 and self.ComputeArray[2][2]==0 :
                pass 
            
            elif self.ComputeArray[0][0] ==0:
                self.ComputeArray[0][0]=-1
                self.history.append([0,0])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
   
                return self.ComputeArray,1
            elif self.ComputeArray[1][1] ==0:
                self.ComputeArray[1][1]=-1
                self.history.append([1,1])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                return self.ComputeArray,1
            elif self.ComputeArray[2][2] ==0:
                self.ComputeArray[2][2]=-1
                self.history.append([2,2])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                return self.ComputeArray,1

        if((self.ComputeArray[0][2]==-1 and self.ComputeArray[1][1]==-1 and self.ComputeArray[2][0]==0) or ( self.ComputeArray[0][2]==-1 and self.ComputeArray[2][0]==-1 and self.ComputeArray[1][1]==0) or ( self.ComputeArray[1][1]==-1 and self.ComputeArray[2][0]==-1 and self.ComputeArray[0][2]==0)):
            if self.ComputeArray[0][2]==0 and self.ComputeArray[1][1]==0 and self.ComputeArray[2][0]==0 :
                pass 
            elif self.ComputeArray[0][2] ==0:
                self.ComputeArray[0][2]=-1
                self.history.append([0,2])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                return self.ComputeArray,1
            elif self.ComputeArray[1][1] ==0:
                self.ComputeArray[1][1]=-1
                self.history.append([1,1])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                return self.ComputeArray,1
            elif self.ComputeArray[2][0] ==0:
                self.ComputeArray[2][0]=-1
                self.history.append([2,0])
                self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                return self.ComputeArray,1
        
            
        prob=random.randint(0,1)
        for x in horiSums:
            if x == 3:
                self.ComputeArray,2
            if x == -2:
                prob = 0
        for x in vertiSums:
            if x == 3:
                self.ComputeArray,2
            if x == -2:
                prob = 1
        if(prob==0):    
            for x in range(len(horiSums)):
                
                if horiSums[x] == -2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([x,y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray , 1
                if horiSums[x] == -1 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2 and horiSums[0]!=2 and horiSums[1]!=2 and horiSums[2]!=2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            self.history.append([x,y])
                            return self.ComputeArray,0
                if horiSums[x] == 0 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2 and horiSums[0]!=2 and horiSums[1]!=2 and horiSums[2]!=2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([x,y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray,0

                if horiSums[x] >= 2 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2:
                    for y in range(len(self.ComputeArray[x])):
                        self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                    self.history.append([x,y])
                    self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                    return self.ComputeArray,0

            self.ComputeArray = self.ComputeArray.T
            for x in range(len(vertiSums)):
                
                if vertiSums[x] == -2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,1
                if vertiSums[x] == -1 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2 and vertiSums[0] != 2 and vertiSums[1] != 2 and vertiSums[2] != 2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,0
                if vertiSums[x] == 0 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2 and vertiSums[0] != 2 and vertiSums[1] != 2 and vertiSums[2] != 2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,0

                if vertiSums[x] >= 2 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2:
                    for y in range(len(self.ComputeArray[x])):
                        self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                    self.history.append([y,x])
                    self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                    return self.ComputeArray.T,0
            self.ComputeArray = self.ComputeArray.T
            
        if(prob==1):    
            self.ComputeArray = self.ComputeArray.T
            for x in range(len(vertiSums)): 
                if vertiSums[x] == -2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,1
                if vertiSums[x] == -1 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2 and vertiSums[0] != 2 and vertiSums[1] != 2 and vertiSums[2] != 2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,0
                if vertiSums[x] == 0 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2 and vertiSums[0] != 2 and vertiSums[1] != 2 and vertiSums[2] != 2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([y,x])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray.T,0
                if vertiSums[x] >= 2 and vertiSums[0] != -2 and vertiSums[1] != -2 and vertiSums[2] != -2:
                    for y in range(len(self.ComputeArray[x])):
                        self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                    self.history.append([y,x])
                    self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                    return self.ComputeArray.T,0
            self.ComputeArray = self.ComputeArray.T
            for x in range(len(horiSums)):
                if horiSums[x] == -2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([x,y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray , 1
                if horiSums[x] == -1 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2 and horiSums[0]!=2 and horiSums[1]!=2 and horiSums[2]!=2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([x,y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray,0
                if horiSums[x] == 0 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2 and horiSums[0]!=2 and horiSums[1]!=2 and horiSums[2]!=2:
                    for y in range(len(self.ComputeArray[x])):
                        if self.ComputeArray[x][y] == 0:
                            self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                            self.history.append([x,y])
                            self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                            return self.ComputeArray,0

                if horiSums[x] >= 2 and horiSums[0]!=-2 and horiSums[1]!=-2 and horiSums[2]!=-2:
                    for y in range(len(self.ComputeArray[x])):
                        self.ComputeArray[x][y] = self.convertor(self.ComputeArray[x][y])
                    self.history.append([x,y])
                    self.puzzleHistory=np.append(self.puzzleHistory, np.reshape(self.ComputeArray,(1,9)),axis=0)                
                    return self.ComputeArray,0
        return self.ComputeArray,0

    def convertor(self,x,zero=None):
        if x == 1:
            return 1
        if x == -1:
            return -1
        if x == 0:
            return -1

while True:
    os.system("clear")
    try:
        print("You are +1 and Computer is -1\n")
        game = TTT()
        while True:
            x,y = input("Enter your move x y : \n").split(" ")
            print(f"you marked at {x} and {y}.\n")
            game.move([int(x),int(y)])
            comp,res = game.predict()
            print(comp)
            
            
            if res==2:
                print("Player Wins, Program Lost(")
                q = input("press enter to continue\n")
                raise Exception("a")


            if res==0:
                pass
            if res==1:
                
                print("Game Over Computer Wins")
                q = input("press enter to continue\n")
                raise Exception("a")
            if len(game.moveRec())==9:
                print("Game Draw")

                q = input("press enter to continue\n")
                raise Exception("a")
    except Exception as e:
        
        Y_append = game.moveRec()
        X_append = game.puzzleRec()
        X_append = X_append[:-1]
        X = np.loadtxt('X')
        Y = np.loadtxt('Y',delimiter=",")

        print(X.shape)
        print(X_append.shape)
        X = np.append(X,X_append,axis=0)
        Y = np.append(Y,Y_append,axis=0)
        np.savetxt('X',X)
        np.savetxt('Y',Y,delimiter=',')
        
        print(X)
        print(Y)
        q = input("press enter to continue\n")
        