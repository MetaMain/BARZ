#Code for attacking the BARZ network
import torch
import NetworkConstructorsAdaptive
import DataManagerPytorch as DMP
import AttackWrappersAdaptiveBlackBox
import BarrierZoneDefense
import ResNetBARZ
import ModelPlus
import BarrierZoneDefenseRandomized

#Run the adaptive/mixed black-box attack on BARZ-8 for the CIFAR-10 dataset
def AdaptiveAttackBARZ8(modelDir):
    saveTag ="Adaptive Attack BARZ-8R6"
    device = torch.device("cuda")    
    #Load the defense
    defense = LoadBARZ8(modelDir)
    #Attack parameters 
    numAttackSamples = 1000
    epsForAttacks = 0.05
    clipMin = -0.5 
    clipMax = 0.5
    #Parameters of training the synthetic model 
    imgSize = 32
    batchSize = 128
    numClasses = 10
    numIterations = 4
    epochsPerIteration = 10
    epsForAug = 0.1 #when generating synthetic data, this value is eps for FGSM used to generate synthetic data
    learningRate = 0.0001 #Learning rate of the synthetic model 
    #Load the training dataset, validation dataset and the defense 
    trainLoader = DMP.GetCIFAR10Training(imgSize, batchSize, mean=0.5)
    valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize, mean=0.5)
    #Get the clean data 
    xTest, yTest = DMP.DataLoaderToTensor(valLoader)
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
    #Create the synthetic model 
    syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    syntheticModel.to(device)
    #Do the attack 
    oracle = defense
    dataLoaderForTraining = trainLoader
    optimizerName = "adam"
    #Last line does the attack 
    AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, cleanLoader, valLoader, numClasses, epsForAttacks, clipMin, clipMax)

#Load the 8 network BARZ defense
def LoadBARZ8(modelDir):
    #List to hold the models 
    modelListPlus = []
    #Model parameters 
    numClasses = 10
    inputImageSize = 32
    batchSize = 32
    threshold = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaleFactors = [1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 3.0, 3.25]
    #Load all 8 BARZ models
    for i in range(0, len(scaleFactors)):
        model = ResNetBARZ.resnet56(32, 0.0, scaleFactors[i], numClasses) 
        checkpoint = torch.load(modelDir+"ModelResNet56BARZ-"+str(scaleFactors[i])+".th")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        #Append the current model to the model list 
        modelListPlus.append(ModelPlus.ModelPlus("ModelResNet56BARZ-"+str(scaleFactors[i]), model, device, inputImageSize, inputImageSize, batchSize))
    #Call the constructor for the BARZ defense 
    defenseBARZ = BarrierZoneDefense.BarrierZoneDefense(modelListPlus, numClasses, threshold)
    return defenseBARZ

#Load the 8 network BARZ defense
def LoadBARZ8Randomized(modelDir):
    #List to hold the models 
    modelListPlus = []
    #Model parameters 
    numClasses = 10
    inputImageSize = 32
    batchSize = 32
    threshold = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaleFactors = [1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 3.0, 3.25]
    #Load all 8 BARZ models
    for i in range(0, len(scaleFactors)):
        model = ResNetBARZ.resnet56(32, 0.0, scaleFactors[i], numClasses) 
        checkpoint = torch.load(modelDir+"ModelResNet56BARZ-"+str(scaleFactors[i])+".th")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        #Append the current model to the model list 
        modelListPlus.append(ModelPlus.ModelPlus("ModelResNet56BARZ-"+str(scaleFactors[i]), model, device, inputImageSize, inputImageSize, batchSize))
    #Call the constructor for the BARZ defense 
    defenseBARZ = BarrierZoneDefense.BarrierZoneDefense(modelListPlus, numClasses, threshold)
    return defenseBARZ

def TestCleanAcc(modelDir):
    defense = LoadBARZ8(modelDir)
    valLoader = DMP.GetCIFAR10Validation()
    score = defense.validateD(valLoader)
    print("Score:", score)
