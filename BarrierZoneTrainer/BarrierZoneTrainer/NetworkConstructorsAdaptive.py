#Network constructors for the adaptive black-box attack 
import torch.nn
import torch.nn.functional as F

class CarliniNetwork(torch.nn.Module):
    def __init__(self, inputImageSize, numClasses=10):
        super(CarliniNetwork, self).__init__()
        #Parameters for the network 
        params=[64, 64, 128, 128, 256, 256]
        #Create the layers 
        #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape))
        #model.add(Activation('relu'))
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=params[0], kernel_size = (3,3), stride = 1)
        #model.add(Conv2D(params[1], (3, 3)))
        #model.add(Activation('relu'))
        self.conv1 = torch.nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size = (3,3), stride = 1)
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        self.mp0 = torch.nn.MaxPool2d(kernel_size=(2,2))
        #model.add(Conv2D(params[2], (3, 3)))
        #model.add(Activation('relu'))
        self.conv2 = torch.nn.Conv2d(in_channels=params[1], out_channels=params[2], kernel_size = (3,3), stride = 1)
        #model.add(Conv2D(params[3], (3, 3)))
        #model.add(Activation('relu'))
        self.conv3 = torch.nn.Conv2d(in_channels=params[2], out_channels=params[3], kernel_size = (3,3), stride = 1)
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2,2))
        #model.add(Flatten())
        #model.add(Dense(params[4]))
        #model.add(Activation('relu'))
        #Next is flatten but we don't know the dimension size yet so must compute
        tetsInput = torch.zeros((1, 3, inputImageSize, inputImageSize))
        outputShape = self.figureOutFlattenShape(tetsInput)
        self.forward0 = torch.nn.Linear(in_features=outputShape[1], out_features=params[4]) #fix later 
        #model.add(Dropout(0.5))
        self.drop0 = torch.nn.Dropout(0.5)
        #model.add(Dense(params[5]))
        #model.add(Activation('relu'))
        self.forward1 = torch.nn.Linear(in_features=params[4], out_features=params[5])
        #model.add(Dense(numClasses, name="dense_2"))
        #model.add(Activation('softmax'))
        self.forward2 = torch.nn.Linear(in_features=params[5], out_features=numClasses)

    def forward(self, x):
        out = F.relu(self.conv0(x)) #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape)) #model.add(Activation('relu'))
        out = F.relu(self.conv1(out)) #model.add(Conv2D(params[1], (3, 3))) #model.add(Activation('relu'))
        out = self.mp0(out)  #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = F.relu(self.conv2(out)) #model.add(Conv2D(params[2], (3, 3)))  #model.add(Activation('relu'))
        out = F.relu(self.conv3(out)) #model.add(Conv2D(params[3], (3, 3))) #model.add(Activation('relu'))
        out = self.mp1(out) #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = out.view(out.size(0), -1) #model.add(Flatten())
        out =  F.relu(self.forward0(out)) #model.add(Dense(params[4])) #model.add(Activation('relu'))
        out = self.drop0(out) #model.add(Dropout(0.5))
        out = F.relu(self.forward1(out)) #model.add(Dense(params[5])) #model.add(Activation('relu'))
        out = F.softmax(self.forward2(out)) #model.add(Dense(numClasses, name="dense_2")) #model.add(Activation('softmax'))
        return out

    #This method is used to figure out what the input to the feedfoward part of the network should be 
    #We have to do this because Pytorch decided not to give this built in functionality for some reason 
    def figureOutFlattenShape(self, x):
        out = F.relu(self.conv0(x)) #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape)) #model.add(Activation('relu'))
        out = F.relu(self.conv1(out)) #model.add(Conv2D(params[1], (3, 3))) #model.add(Activation('relu'))
        out = self.mp0(out)  #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = F.relu(self.conv2(out)) #model.add(Conv2D(params[2], (3, 3)))  #model.add(Activation('relu'))
        out = F.relu(self.conv3(out)) #model.add(Conv2D(params[3], (3, 3))) #model.add(Activation('relu'))
        out = self.mp1(out) #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = out.view(out.size(0), -1) #model.add(Flatten())
        return out.shape
