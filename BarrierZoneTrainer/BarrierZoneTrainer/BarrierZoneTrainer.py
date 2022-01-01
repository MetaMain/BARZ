import torch
import os
import TrainingMethods
import AttackMethods
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Main function 
def main():
    #Set the directory where the BARZ models are saved, e.g. modelDir = "C://Users//Downloads//BARZ-8 PyTorch Models//"
    modelDir = ""
    #This runs the adaptive (mixed) black-box attack on the BARZ-8 Defense 
    AttackMethods.AdaptiveAttackBARZ8(modelDir)

    #Uncomment lines 16 through 19 to train your own BARZ network from scratch on CIFAR-10 
    #Parameters for BARZ CIFAR-10 defense
    #percentBiasPixels = 0.35 #What percent of the pixels should be manipulated in the defense 
    #scaleFactor = 1.0 #Resize factor for the defense, 1.0 means no resizing 
    #saveTag = "ResNet56BARZ-"+str(scaleFactor) #Name for saving the model 
    #TrainingMethods.TrainBARZNetwork(percentBiasPixels, scaleFactor, saveTag)

if __name__ == '__main__':
    main()


