#Code for training the BARZ network
import torch 
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import ResNetBARZ
import time
import os

#Train the BARZ network from scratch 
def TrainBARZNetwork(percentBiasPixels, scaleFactor, saveTag):
    #Create the BARZ model for CIFAR-10
    model = ResNetBARZ.resnet56(inputImageSize=32, percentBiasPixels=percentBiasPixels, scaleFactor=scaleFactor, numClasses=10)    
    #Training parameters
    numEpochs = 100
    batchSize = 32#128
    learningRate = 0.01#0.1
    momentum = 0.9
    printFreq = 100 #print frequency 
    saveEvery = 10
    weightDecay = 2e-4
    workers = 4
    #End training parameters 
    bestAcc = 0
    valBatchSize = batchSize 
    #Get the current directory 
    saveDir = os.getcwd() + "//"+saveTag+"//"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    #Get the device and put the model on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Get the dataset and start the training 
    transformTrain = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
    transformTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, transform=transformTrain, download=True), batch_size=batchSize, shuffle=True, num_workers=workers, pin_memory=True)
    #pin memory = true allows data to be loaded on the CPU and the further loaded onto the GPU during training, speeds up transfer between host and device
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transformTest), batch_size=valBatchSize, shuffle=False, num_workers=workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
    lrScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120, 140, 160, 180], gamma=1e-1)
    #Train for a number of epochs 
    for epoch in range(0, numEpochs):
        #train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(trainLoader, model, criterion, optimizer, epoch, printFreq, device)
        lrScheduler.step()
        #evaluate on validation set
        currentAcc = validate(valLoader, model, criterion)
        #remember best prec@1 and save checkpoint
        #If this is the best accuracy then save the model 
        if currentAcc >= bestAcc:
            bestAcc = currentAcc
            saveCheckpoint({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'bestAcc': bestAcc,
                }, filename=os.path.join(saveDir, 'Model'+saveTag+'.th'))

def train(train_loader, model, criterion, optimizer, epoch, printFreq, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
   
        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        #if args.half:
        #    input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % printFreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            #if args.half:
            #    input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    print('Test: [{0}/{1}]\t'
            #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #              i, len(val_loader), batch_time=batch_time, loss=losses,
            #              top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

#save the trained model 
def saveCheckpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res