# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
# :: as separator is used because to need to separate movies
movies = pd.read_csv("ml-1m/movies.dat", sep="::", header=None, engine="python", encoding="latin-1")
users = pd.read_csv("ml-1m/users.dat", sep="::", header=None, engine="python", encoding="latin-1")
rating = pd.read_csv("ml-1m/ratings.dat", sep="::", header=None, engine="python", encoding="latin-1")
print(rating)
# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype="int")
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype="int")

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
# To create a structure of node that RBM excepts as input
# Observations in line and features in column
# Cant do 2D numpy array as we are using pytorch
# SO we will create a list of list( Several lists)
# One list for one user
# if user didn't rate a movie we will give that movie 0
# so we will have same size for each of training and test sets
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# Tensors are arrays that contain data of single datatype. This is a PyTorch arrays
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
# Becuase we have to predict in binary format also RBM will take input vector also input = output
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
# Pytorch doesnt accept single D input hence we create 2D wherever required
class RBM:
    # nv = no. visible nodes, nh = no. of hidden nodes
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        # a, b constants remember the formula

    # Now lets do sampling the hidden nodes according to probablilty, it will activate them
    # x= visible nodes as per probability
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        # if user has 1 for all drama movies so the probability of that node corresponding to drama movies will be high
        p_h_given_v = torch.sigmoid(activation)
        # bernoulli activates the nodes as per the nodes going over a threshold and get random no. b/w 0 and 1
        # if its greater than the node is activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        # if user has 1 for all drama movies so the probability of that node corresponding to drama movies will be high
        p_v_given_h = torch.sigmoid(activation)
        # bernoulli activates the nodes as per the nodes going over a threshold and get random no. b/w 0 and 1
        # if its less than the user will like but all this depends on the activation of its respective hidden node
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # vk = k visible nodes and phk is probability after p h given vk
    # vo =  rating given by user initially ph0 is probab after input vector according to v0
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users-batch_size, batch_size):
        # output of gibbs sampling the input node gets updated after each round trip
        vk = training_set[id_user:id_user+batch_size]
        # original ratings that we wont update or touch since we need to check the error
        v0 = training_set[id_user:id_user + batch_size]
        # , _ returns only first return value
        ph0, _ = rbm.sample_h(v0)
        # for loop for k steps of contrastive divergence
        for k in range(10):
            # First sampling of hidden node
            _, hk = rbm.sample_h(vk)
            # Samples visible node after the 1st step of gibbs sampling
            _, vk = rbm.sample_v(hk)
            # we freeze the visible nodes that didn't have no ratings
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        # Exclude the ratings that weren't present
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    # It will activate the hidden neuron
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
