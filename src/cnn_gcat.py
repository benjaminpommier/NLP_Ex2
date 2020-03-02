# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:31:46 2020

@author: Benjamin Pommier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, embedding, aspect_embedding, Co=100, Ks=[3,4,5], ):
        super(CNN_Gate_Aspect_Text, self).__init__()
        
        embed_num, embed_dim = embedding.shape
        aspect_embed_num, aspect_embed_dim = aspect_embed.shape #Be clear on what is aspect_embed ? Embedding of the different aspect term/class ? 
            # ATTENTION: identify whether aspect_embed_num must be equal to 12 in our case 
            # as this is the number of aspects in our dataset
        
        C = 3 #Number of classes to predict. Here constant to 3.

        # Co = kernel_num #Number of outchannels
        # Ks = kernel_sizes #Size of the different kernels

        #Initialize the embedding, with weights if pre-trained embedding provided
        self.embed = nn.Embedding(embed_num, embed_dim) 
        self.embed.weight = nn.Parameter(embedding, requires_grad=True) #What is exactly the type of embedding ?
        
        #Initialise the embedding for the aspect, with weights if pretrained embedding provided
        self.aspect_embed = nn.Embedding(aspect_embed_num, aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList(nn.Conv1d(D, 300, 3, padding=1))
        # self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]]) #old

        self.dropout = nn.Dropout(0.2)
        
        #Predict the classes
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)


    def forward(self, feature, aspect):
        #Aspect embeddings >> TO CHECK: for me, they call aspect, the term related to the aspect category
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)

        #Embedding of the context
        feature = self.embed(feature)  # (N, L, D)
        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]
        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit, x, y