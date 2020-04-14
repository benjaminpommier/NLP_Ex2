#!/usr/bin/env python
# coding: utf-8

# # ATSA Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from time import time


# ## Loading data

# In[2]:


path_data = "./data/"
path_src = "./src/"

# train set
df_train = pd.read_csv(path_data + 'traindata.csv', sep = '\t', header = None)
df_train.columns = ['polarity', 'aspect_category', 'target_term', 'start:end', 'sentence']

#dev set
df_dev = pd.read_csv(path_data + 'devdata.csv', sep = '\t', header = None)
df_dev.columns = ['polarity', 'aspect_category', 'target_term', 'start:end', 'sentence']

df_train.head(5)


# ## Création de y_train et y_dev

# In[3]:


y_train = torch.Tensor(df_train['polarity'].map({'positive':2, 'neutral':1, 'negative':0}).values)
y_dev = torch.Tensor(df_dev['polarity'].map({'positive':2, 'neutral':1, 'negative':0}).values)

y_train.shape


# ## Création de X_train et X_dev

# In[4]:


vocab_size = 20000

sentences_train = df_train['sentence']
sentences_dev = df_dev['sentence']
sentences = pd.concat([sentences_train, sentences_dev])

sentences = list(sentences.apply(lambda sentence: one_hot(sentence, vocab_size, lower=False)).values)
X1 = torch.LongTensor(pad_sequences(sentences))
X1.shape


# In[5]:


vocab_size_target = 2000

target_train = df_train['target_term']
target_dev = df_dev['target_term']
targets = pd.concat([target_train, target_dev])

targets = list(targets.apply(lambda sentence: one_hot(sentence, vocab_size_target, lower=False)).values)
X2 = torch.LongTensor(pad_sequences(targets))
X2.shape


# ## After embedding

# In[6]:


# max number for context and aspect
max_aspect = 2
max_context = 30

# useful params
l1 = min(X1.shape[1], max_context) # max length of a sentence
l2 = min(X2.shape[1], max_aspect) # max length of target name
train_size = int(X1.shape[0] * 0.8) # take 80% of data for train set and 20% for dev set

# reduce dimension
X1 = X1[:,-min(l1,max_context):]
X2 = X2[:,-min(l2,max_aspect):]

# gather tensor
X = torch.cat([X1, X2], 1)

# train set & dev set creation
X_train = X[:train_size, :]
X_dev = X[train_size:, :]

print(X_train.shape)
print(X_dev.shape)


# In[7]:


dataset_train = TensorDataset(X_train, y_train)
dataset_dev = TensorDataset(X_dev, y_dev)


# In[8]:


print('Train set')
print(pd.Series(y_train).value_counts(normalize = True))
print('')
print('Dev set')
print(pd.Series(y_dev).value_counts(normalize = True))


# ## Model

# In[9]:


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, Co=100, L=300, Ks=[3,4,5], C=3, embed_num = 20000, embed_dim = 300, aspect_embed_num = 2000, aspect_embed_dim = 300, embedding = None, aspect_embedding = None):
        super(CNN_Gate_Aspect_Text, self).__init__()
        #Initialize the embedding, with weights if pre-trained embedding provided
        self.embed = nn.Embedding(embed_num, embed_dim) 
        # self.embed.weight = nn.Parameter(embedding, requires_grad=True) #What is exactly embedding ?
        
        #Initialise the embedding for the aspect, with weights if pretrained embedding provided
        self.aspect_embed = nn.Embedding(aspect_embed_num, aspect_embed_dim)
        # self.aspect_embed.weight = nn.Parameter(aspect_embedding, requires_grad=True)

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
        x = [torch.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [torch.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]
        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1) #Check what is it ?
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit


# ## def train

# In[95]:


# Create the model: 
model = CNN_Gate_Aspect_Text()

# Hyperparameters for training: 
num_epochs = 10
batch_size = 32
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-0, weight_decay=0.01)


# In[96]:


# Calculate the accuracy to evaluate the model
def accuracy(dataset, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        dataloader = DataLoader(dataset)
        for X, labels in dataloader:
            outputs = model(X[:, :l1], X[:, -l2:])
            _, predicted = torch.max(outputs.data, 1) 
            correct += (predicted == labels).sum()

    return 100*correct.item()/ len(dataset)


# In[97]:


# define a function for training
def train(model, dataset_train, dataset_dev, num_epochs, batch_size, criterion, optimizer):
    t = time()
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        acc = 0.
        for (X_batch, labels) in train_loader:
            y_pre = model(X_batch[:, :l1], X_batch[:, -l2:])
            loss = criterion(y_pre, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_pre.data, 1) 
            acc += (predicted == labels).sum().item()
        
        acc = 100 * acc / len(dataset_train)
        dev_acc = accuracy(dataset_dev, model)
        print('Epoch [{}/{}] | exec time: {:.2f} secs | acc: {:.2f}% | dev_acc: {:.2f}%'.format(epoch+1, num_epochs, time()-t, acc, dev_acc))


# In[98]:


train(model, dataset_train, dataset_dev, num_epochs, batch_size, criterion, optimizer)


# ## Evaluation on dev set

# In[92]:


from sklearn.metrics import classification_report

def report(dataset, model):
    predicted_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        correct = 0
        dataloader = DataLoader(dataset)
        for X, labels in dataloader:
            outputs = model(X[:, :l1], X[:, -l2:])
            _, predicted = torch.max(outputs.data, 1) 
            correct += (predicted == labels).sum()
            predicted_all.append(int(predicted[0]))
            labels_all.append(int(labels[0]))

    print(classification_report(labels_all,predicted_all))


# In[93]:


# Dev set
accuracy_dev = accuracy(dataset_dev, model)
print('Accuracy for dev set is : {:.2f} %'.format(accuracy_dev))
print('')
report(dataset_dev, model)


# In[94]:


# train set
accuracy_train = accuracy(dataset_train, model)
print('Accuracy for train set is : {:.2f} %'.format(accuracy_train))
print('')
report(dataset_train, model)


# In[ ]:




