import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas

num_of_data = 10000
epochs = 3
max_random_num = 50
sequence_size = 100 # you should want to change the NN architecture if you change this
lr = 0.001
seed = 33
loss_print_frequency = 100 # every [x] times the model has been trained, it will produce the average loss
number_of_tests = 1000
hidden_state = 30
num_of_layers = 3

random.seed(seed)

def one_hot_encoding (num, length):
    array = [0 for i in range(length)]
    array[num - 1] = 1
    return array

def decode_encoding (array):
    return torch.argmax(array)

def generate_data ():
    x_train = []
    y_train = []
    for i in range(num_of_data):
        array = [random.randint(1,max_random_num) for i in range(sequence_size + 1)]
        x = [one_hot_encoding(array[i], max_random_num) for i in range(sequence_size)]
        y = one_hot_encoding(array[sequence_size], max_random_num)
        
        x_train.append(x)
        y_train.append(y)
    return np.array(x_train).reshape((num_of_data, sequence_size * max_random_num)), np.array(y_train)

x_train, y_train = generate_data()

print(x_train.shape) # will output (10000, 5, 10)
print(y_train.shape) # will output (10000, 1, 10)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 100)
        self.fc2 = nn.Linear(100, y_train.shape[1])
        if torch.cuda.is_available ():
            self.device = "cuda"
            print("Running on the GPU")
        else:
            self.device = "cpu"
            print("Running on the CPU")
        self.to(self.device)

    def forward (self, x): 
        x = self.fc1(torch.from_numpy(x).float().to(self.device)) 
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x.to("cpu")

neuralNet = Net()

lossNN = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neuralNet.parameters(), lr=lr)

for epoch in range(1, epochs+1):
    total_loss = 0
    for iteration in range(num_of_data):
        # get data and train
        rand_num = random.randint(0, num_of_data-1)
        x = x_train[rand_num]
        y = y_train[rand_num]
        optimizer.zero_grad()
        output = neuralNet(x)
        output = output.reshape(1, output.shape[0])
        y = torch.from_numpy(np.array([np.argmax(y)]))
        loss = lossNN(output, y)
        loss.backward()
        optimizer.step()
        
        # calc loss
        total_loss += loss
        if iteration % loss_print_frequency == 0:
            print(f"Loss: {total_loss / loss_print_frequency}")
            total_loss = 0
    print(f"Epoch: {epoch}")

# test

correct = 0

for i in range(number_of_tests):
    array = [random.randint(1,max_random_num) for i in range(sequence_size + 1)]
    x = np.array([one_hot_encoding(array[i], max_random_num) for i in range(sequence_size)]).reshape((max_random_num * sequence_size))
    if decode_encoding(neuralNet(x)) == array[sequence_size]:
        correct += 1 

print(f"Percentage correct: {correct / float(number_of_tests)}")
