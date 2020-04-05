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