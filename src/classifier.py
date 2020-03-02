__authors__ = ['']

#Importing modules
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np


class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """



def train(train_iter, dev_iter, model, args, epoch=5):
    # python -m run -lr 5e-3 -batch-size 32 -verbose 1 -model CNN_Gate_Aspect -embed_file glove -r_l r -year 14 -epochs 6 -atsa

    time_stamps = []

    optimizer = torch.optim.Adagrad(model.parameters()) #Change

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc = 0
    for epoch in range(1, epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment

            feature.data.t_()
            if len(feature) < 2:
                continue
            aspect.data.unsqueeze_(0)
            aspect.data.t_()
            target.data.sub_(1)  # batch first, index align

            optimizer.zero_grad()
            logit, _, _ = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            loss.backward()

            optimizer.step()

            steps += 1
            if steps % 10 == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.data[0],
                                                                                 accuracy,
                                                                                 corrects,
                                                                                 batch.batch_size))

        if epoch == epochs:
            dev_acc, _, _ = eval(dev_iter, model)

            delta_time = time.time() - start_time
            print('\n{:.4f} - {:.4f} - {:.4f}'.format(dev_acc, delta_time))
            time_stamps.append((dev_acc, delta_time))
            print()
            
    return dev_acc, time_stamps


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    loss = None
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature.data.t_()
        aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align

        logit, pooling_input, relu_weights = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size # Should be avg_loss instead of loss.data[0] ?
    accuracy = 100.0 * corrects/size
    model.train()
    
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size))
    
    return accuracy, pooling_input, relu_weights