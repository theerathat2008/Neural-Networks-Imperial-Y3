import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier

class Net(nn.Module):
    def __init__(self, input_size, hiddenLayer1_size, hiddenLayer2_size, output_size):
        super(Net, self).__init__()
        self.full_connect1 = nn.Linear(input_size, hiddenLayer1_size)
        self.full_connect2 = nn.Linear(hiddenLayer1_size, hiddenLayer2_size)
        self.full_connect3 = nn.Linear(hiddenLayer2_size, output_size)


    def forward(self, x):
        x = F.relu(self.full_connect1(x))
        x = F.relu(self.full_connect2(x))
        x = F.sigmoid(self.full_connect3(x))
        return x

# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, epoch_num, learning_rate, batch_size, model = None, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        self.base_classifier = None
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_size = None
        self.hiddenLayer1_size = None
        self.hiddenLayer2_size = None
        self.output_size = None
        self.optimizer = None
        self.criterion = None
        self.x_test = None
        self.y_test = None
        self._model = model


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """

        # Remove NaN values
        X_raw = X_raw.dropna()

        # Convert categorical data to label encoding
        columns = ['id_policy', 'pol_coverage', 'pol_pay_freq', 'pol_payd', 'pol_usage', 'pol_insee_code', 'drv_drv2',
                   'drv_sex1', 'drv_sex2', 'vh_fuel',
                   'vh_make', 'vh_model', 'vh_type', 'regional_department_code']
        self._labelEncode(X_raw, columns)

        sc = StandardScaler()
        sc.fit(X_raw)
        data = sc.transform(X_raw)

        return data

    def _labelEncode(self, data, columns):
        labelEncoder = LabelEncoder()

        for column in columns:
            data[column] = labelEncoder.fit_transform(data[column])

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.severityModel = stats.mode(claims_raw[nnz])

        X_clean = self._preprocessor(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_raw, test_size=0.2)

        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(np.array(y_train))

        tensorData = torch.utils.data.TensorDataset(X_train, y_train)

        trainloader = torch.utils.data.DataLoader(tensorData, batch_size=self.batch_size)

        self.x_test, self.y_test = X_test, y_test

        input_size, hiddenLayer1_size, hiddenLayer2_size, output_size = 35, 100, 100, 1
        model = Net(input_size, hiddenLayer1_size, hiddenLayer2_size, output_size)
        self.input_size, self.hiddenLayer1_size, self.hiddenLayer2_size, self.output_size = input_size, hiddenLayer1_size, hiddenLayer2_size, output_size
        self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
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

        self.save_model("part3_pricing_model.pickle")

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        x_clean = self._preprocessor(X_raw)

        x_tensor = torch.from_numpy(x_clean).float()
        dataLoader = torch.utils.data.DataLoader(x_tensor, batch_size=1)

        model = self._model
        y_pred = []
        y_binary = []
        with torch.no_grad():
            for data in dataLoader:
                outputs = model(data)

                y_pred.append(outputs[0][0].item())

                if (outputs[0][0].item() > 0):
                    y_binary.append(1)
                else:
                    y_binary.append(0)

        return (y_pred, y_binary)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # We take the mode of all prices and use that as a severity model

        return self.predict_claim_probability(X_raw) * self.y_mean

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

epoch_num = 200
batch_size = 100
learning_rate = 0.05