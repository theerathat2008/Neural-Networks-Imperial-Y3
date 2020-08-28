import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pandas as pd

class Net(nn.Module):
    def __init__(self, input_size, hiddenLayer1_size, hiddenLayer2_size, output_size):
        super(Net, self).__init__()
        self.full_connect1 = nn.Linear(input_size, hiddenLayer1_size)
        self.full_connect2 = nn.Linear(hiddenLayer1_size, hiddenLayer2_size)
        self.full_connect3 = nn.Linear(hiddenLayer2_size, output_size)


    def forward(self, x):
        x = F.relu(self.full_connect1(x))
        x = F.relu(self.full_connect2(x))
        x = self.full_connect3(x)
        return x

class ClaimClassifier():

    def __init__(self, epoch_num, learning_rate, batch_size, model = None):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_size = None
        self.hiddenLayer1_size = None
        self.hiddenLayer2_size = None
        self.output_size  = None
        self.optimizer = None
        self.criterion = None
        self.x_test = None
        self.y_test = None
        self._model = model

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """

        sc = StandardScaler()
        sc.fit(X_raw)
        cleandata = sc.transform(X_raw)

        return cleandata

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        cleanData = self._preprocessor(X_raw)
        
    
        X_train, X_test, y_train, y_test = train_test_split(cleanData, y_raw, test_size = 0.2)
       
        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(np.array(y_train))


        tensorData = torch.utils.data.TensorDataset(X_train, y_train)
        
        trainloader = torch.utils.data.DataLoader(tensorData, batch_size = self.batch_size)

    

        self.x_test, self.y_test = X_test, y_test

        input_size, hiddenLayer1_size, hiddenLayer2_size, output_size = 9, 100, 100, 1
        model = Net(input_size, hiddenLayer1_size, hiddenLayer2_size, output_size)
        self.input_size, self.hiddenLayer1_size, self.hiddenLayer2_size, self.output_size = input_size, hiddenLayer1_size, hiddenLayer2_size, output_size
        self.optimizer = optim.SGD(model.parameters(), lr = self.learning_rate)
        self.criterion = nn.L1Loss()

        for epoch in range(self.epoch_num):

            model.train()
            train_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                attributes, labels = data
                
                
                attributes = attributes.reshape(-1, self.input_size)
                
                outputs = model(attributes.float())
                
                
                outputs = np.squeeze(outputs, axis=1)

                loss = self.criterion(outputs, labels.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            print(train_loss)

        self._model = model

        self.save_model("model1 .pickle")


    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        print(X_raw)
        x_clean = self._preprocessor(X_raw)

        x_tensor = torch.from_numpy(x_clean).float()
        dataLoader = torch.utils.data.DataLoader(x_tensor, batch_size = 1)
        
        model = self._model
        y_pred = []
        with torch.no_grad():
            for data in dataLoader:

                outputs = model(data)
                
                
                if(outputs[0][0].item() > 0):
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        return y_pred

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        y_pred = self.predict(self.x_test)
        correct = 0
        # for i in range(100):
        #     if(self.y_test[i] == 1):
        #         print(y_pred[i])
        
        # print(self.y_test[0:100])
        for i in range(len(y_pred)):
            
            if(y_pred[i] == self.y_test[i]):
                correct += 1

        print("Accuracy:" + str(100*(correct/len(y_pred))))


    def save_model(self, path):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open(path, 'wb') as target:
            pickle.dump(self, target)


def load_model(path):
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open(path, 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters

#Set hyper-parameters

epoch_num = 200
batch_size = 100
learning_rate = 0.05

claim_classifier = load_model("part2_claim_classifier")

data = pd.read_csv("part2_training_data.csv")
train_data = data.drop(["claim_amount", "made_claim"], axis=1).to_numpy()
label_data = data["made_claim"].to_numpy()
