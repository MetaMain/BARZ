import numpy
import random
import DataManagerPytorch as DMP

class BarrierZoneDefense():
    #Default constructor 
    def __init__(self , modelPlusList, classNum, threshold):
        self.ModelPlusList=modelPlusList
        self.ClassNum=classNum
        self.ModelNum=len(self.ModelPlusList)
        self.Threshold=threshold

    #Majority voting AND thresholding 
    def predictD(self, dataLoader, numClasses):
        #basic error checking
        if numClasses != self.ClassNum:
            raise ValueError("Class numbers don't match for BARZ defense")
        sampleSize = len(dataLoader.dataset) 
        modelVotes=numpy.zeros((self.ModelNum, sampleSize, self.ClassNum)) 

        #Get the votes for each of the networks 
        for i in range(0,self.ModelNum):
            print("Evaluating on model:", self.ModelPlusList[i].modelName)
            modelVotes[i,:,:]=self.ModelPlusList[i].predictD(dataLoader, self.ClassNum)
        
        #Now do the voting 
        finalVotes=numpy.zeros((sampleSize,self.ClassNum+1)) #The (n+1)th class is the noise class
        for i in range(0, sampleSize):
            currentTally=numpy.zeros((self.ClassNum,))
            for j in range(0, self.ModelNum):
                currentVote=modelVotes[j,i,:].argmax(axis=0)
                currentTally[currentVote]=currentTally[currentVote]+1
            if (currentTally[currentTally.argmax(axis=0)]>=self.Threshold): #Make sure it is above the threshold 
                finalVotes[i,currentTally.argmax(axis=0)]=1.0
            else: #Make it the last "noise" class
                finalVotes[i,self.ClassNum]=1.0
        return finalVotes

    def validateD(self, dataLoader):
        accuracy=0
        sampleSize=len(dataLoader.dataset) 
        multiModelOutput=self.predictD(dataLoader, self.ClassNum)
        xTest, yTest = DMP.DataLoaderToTensor(dataLoader)
        for i in range(0, sampleSize):
            if(multiModelOutput[i].argmax(axis=0)==yTest[i]):
                accuracy=accuracy+1
        accuracy=accuracy/sampleSize
        return accuracy

    #the network is fooled if we don't have a noise class label AND it gets the wrong label 
    #Returns attack success rate 
    def evaluateAdversarialAttackSuccessRate(self, advLoader):
        yPred = self.predictD(advLoader, self.ClassNum)
        xAdv, yCleanSingleVal = DMP.DataLoaderToTensor(advLoader)
        advAcc=0
        for i in range(0, xAdv.shape[0]):
            #The attack wins only if we don't correctly label the sample AND the sample isn't given the nosie class label
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yCleanSingleVal[i]: #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/ float(xAdv.shape[0])
        return advAcc

class BarrierZoneDefenseRandomized():
    #Default constructor 
    def __init__(self , modelPlusList, classNum, modelsPerEval):
        self.ModelPlusList=modelPlusList
        self.ClassNum=classNum
        self.ModelNum=len(self.ModelPlusList)
        self.Threshold=modelsPerEval
        self.ModelIndexList = list(range(0, self.ModelNum))

    #Majority voting AND thresholding 
    def predictD(self, dataLoader, numClasses):
        #basic error checking
        if numClasses != self.ClassNum:
            raise ValueError("Class numbers don't match for BARZ defense")
        sampleSize = len(dataLoader.dataset) 

        #Randomized the list
        random.shuffle(self.ModelIndexList)

        modelVotes=numpy.zeros((self.Threshold, sampleSize, self.ClassNum)) 
        #Get the votes for each of the networks 
        for i in range(0, self.Threshold):
            modelIndex = self.ModelIndexList[i]
            print("Evaluating on model:", self.ModelPlusList[modelIndex].modelName)
            modelVotes[i,:,:]=self.ModelPlusList[modelIndex].predictD(dataLoader, self.ClassNum)
        
        #Now do the voting 
        finalVotes=numpy.zeros((sampleSize,self.ClassNum+1)) #The (n+1)th class is the noise class
        for i in range(0, sampleSize):
            currentTally=numpy.zeros((self.ClassNum,))
            for j in range(0, self.Threshold):
                currentVote=modelVotes[j,i,:].argmax(axis=0)
                currentTally[currentVote]=currentTally[currentVote]+1
            if (currentTally[currentTally.argmax(axis=0)]>=self.Threshold): #Make sure it is above the threshold 
                finalVotes[i,currentTally.argmax(axis=0)]=1.0
            else: #Make it the last "noise" class
                finalVotes[i,self.ClassNum]=1.0
        return finalVotes

    def validateD(self, dataLoader):
        accuracy=0
        sampleSize=len(dataLoader.dataset) 
        multiModelOutput=self.predictD(dataLoader, self.ClassNum)
        xTest, yTest = DMP.DataLoaderToTensor(dataLoader)
        for i in range(0, sampleSize):
            if(multiModelOutput[i].argmax(axis=0)==yTest[i]):
                accuracy=accuracy+1
        accuracy=accuracy/sampleSize
        return accuracy

    #the network is fooled if we don't have a noise class label AND it gets the wrong label 
    #Returns attack success rate 
    def evaluateAdversarialAttackSuccessRate(self, advLoader):
        yPred = self.predictD(advLoader, self.ClassNum)
        xAdv, yCleanSingleVal = DMP.DataLoaderToTensor(advLoader)
        advAcc=0
        for i in range(0, xAdv.shape[0]):
            #The attack wins only if we don't correctly label the sample AND the sample isn't given the nosie class label
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yCleanSingleVal[i]: #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/ float(xAdv.shape[0])
        return advAcc
