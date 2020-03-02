# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:31:46 2020

@author: Benjamin Pommier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, embedding, aspect_embedding, Co=100, L=300, Ks=[3,4,5], C=3, embed_num = 20000, embed_dim = 300, aspect_embed_num = 2000, aspect_embed_dim = 300):
        '''

        Parameters
        ----------
        embedding : TYPE
            DESCRIPTION.
        aspect_embedding : TYPE
            DESCRIPTION.
        Co : TYPE, optional
            DESCRIPTION. The default is 100.
        L : TYPE, optional
            DESCRIPTION. The default is 300.
        Ks : TYPE, optional
            DESCRIPTION. The default is [3,4,5].
        C : TYPE, optional
            DESCRIPTION. The default is 3.
        embed_num : TYPE, optional
            DESCRIPTION. The default is 20000.
        embed_dim : TYPE, optional
            DESCRIPTION. The default is 300.
        aspect_embed_num : TYPE, optional
            DESCRIPTION. The default is 2000.
        aspect_embed_dim : TYPE, optional
            DESCRIPTION. The default is 300.

        Returns
        -------
        None.

        '''
        # super(CNN_Gate_Aspect_Text, self).__init__() # not needed
        
        #Initialize the embedding, with weights if pre-trained embedding provided
        self.embed = nn.Embedding(embed_num, embed_dim) 
        self.embed.weight = nn.Parameter(embedding, requires_grad=True) #What is exactly embedding ?
        
        #Initialise the embedding for the aspect, with weights if pretrained embedding provided
        self.aspect_embed = nn.Embedding(aspect_embed_num, aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(embed_dim, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(embed_dim, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(embed_dim, L, 3, padding=1)])

        self.dropout = nn.Dropout(0.2)
        
        #Predict the classes
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(L, Co)


    def forward(self, feature, aspect):
        #Aspect embeddings >> TO CHECK: for me, they call aspect, the term related to the aspect category
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1) #Check what is it ? Not needed here

        #Embedding of the context
        feature = self.embed(feature)  # (N, L, D)
        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]
        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1) #Check what is it ?
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit