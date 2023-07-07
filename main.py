#  -*-  coding: UTF-8  -*-
# This if a Python script for prediction fake recalls using XGB, CatBoost and DBDT models.
# You can retrain DBDT model on Google Collab at link, also you need to use GPU:
# https://colab.research.google.com/drive/1_nGpL8BLyHjfJXbr4ro9UrKLYj1eb7fU?authuser=1&hl=ru#scrollTo=neMyrXQk1Wqc
# To run this script you can write arguments like examples:
# "python3 main.py input.csv output.csv" - Predictions from input.csv to output.csv file
# "python3 main.py input.csv" - Predictions from input.csv file to default named file output.csv
# "python3 main.py" - Predictions from stdin to stdout


# Import packeges:
import sys
from datetime import datetime
import pandas as pd
import numpy as np

import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(4);

# Class SDTs for DBDT model initialization
class SDTs(nn.Module):
    def __init__(
        self,
        n_x, # Count of input data
        tree_levels=3, # Tree depth
        lmbd=0.1, # Regularization coefficient 
        T=100, # Iterations count
        imratio=None,
        random_seed=4,
        margin=1,
        backend='ce',
        use_cuda=True
        ):
        torch.manual_seed(random_seed) # Init random seed for params init
        super(SDTs, self).__init__()
        self.tree_levels = tree_levels
        self.leafs_numb = 2 ** (self.tree_levels - 1)
        self.inner_numb = self.leafs_numb - 1
        self.lmbd = lmbd
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.T = T
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.backend = 'ce'

        self.V_next_value = torch.zeros([self.inner_numb, self.T])

        self.param_dict = {}
        for i in range(T):
            self.param_dict['W' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(n_x, self.inner_numb)), gain=1))
            self.param_dict['b' + str(i)] = nn.Parameter(
                torch.nn.init.constant_(Variable(torch.randn(1, self.inner_numb)), 0))
            self.param_dict['phi' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(self.leafs_numb, 1), ), gain=1))

            self.params = self.param_dict.values()

    # Functions for implementing iterability by model parameters
    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]

    def node_probability(self, index_node, A):
        p = torch.ones(A.shape[0])
        while index_node - 1 >= 0:
            father_index = int((index_node - 1) / 2)
            if (index_node - 1) % 2 == 0:
                p = p * (1.0 - A[:, father_index])
            else:
                p = p * (A[:, father_index])
            index_node = father_index
        return p

    # Forward Propagation
    def forward_propagation_Boosting(self, X, W, b):
        Z = torch.add(torch.matmul(X, W), b)
        A = torch.sigmoid(1 * Z)
        return A

    # Probability of the input data belonging to a leaf node
    def compute_leafs_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.leafs_numb:
            ta.append(self.node_probability(self.leafs_numb - 1 + i, A))
            i = i + 1
        leafs_prob_matrix = torch.stack(ta, dim=0)
        return leafs_prob_matrix

    # Probability of activating each internal node
    def compute_inner_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.inner_numb:
            ta.append(self.node_probability(i, A))
            i = i + 1
        inner_prob_matrix = torch.stack(ta, dim=0)
        return inner_prob_matrix

    # Loss function regularization exponent
    def compute_regularization(self, A, inner_prob_matrix, V_prec):
        ta = list()
        ema = list()
        i = 0
        while i < self.inner_numb:
            depth = int(np.log(i + 1) / np.log(2))
            decay = 1. - np.exp(-depth)
            a_i = torch.div(torch.matmul(inner_prob_matrix[i, :], A[:, i]), torch.sum(inner_prob_matrix[i, :]))
            w_i = decay * V_prec[i] + (1. - decay) * a_i
            r_i = -self.lmbd * (2 ** (-depth)) * (
                    0.5 * torch.log(w_i) + 0.5 * torch.log(1.0 - w_i))
            ta.append(r_i)
            ema.append(w_i)
            i = i + 1
        regularization = torch.sum(torch.stack(ta, dim=0))
        V_next = torch.stack(ema, dim=0)
        return regularization, V_next

    # Model inference of SDTs by combining the predictions of each tree in the forest
    def compute_Boosting_output(self, X, Y):
        X = X.float()
        output_sum = torch.full_like(Y, Y.mean())
        output = []

        for t in range(self.T):
            A = self.forward_propagation_Boosting(X, self.param_dict['W' + str(t)], self.param_dict['b' + str(t)])
            leafs_prob_matrix = self.compute_leafs_prob_matrix(A)
            inner_prob_matrix = self.compute_inner_prob_matrix(A)
            output.append(torch.matmul(leafs_prob_matrix.permute(1, 0), self.param_dict['phi' + str(t)]))
            output_sum = output_sum + 0.1 * torch.squeeze(output[t])
        return output_sum

    # Method for Calculating the Loss Function
    def compute_cost_Boosting_wr(self, X, Y):
        output_sum = torch.full_like(Y, Y.mean())
        output = []
        V_next = []
        cost_sum = torch.tensor(0)
        Y = Y.reshape(-1)
        for t in range(self.T):
            A = self.forward_propagation_Boosting(X, self.param_dict['W' + str(t)], self.param_dict['b' + str(t)])
            leafs_prob_matrix = self.compute_leafs_prob_matrix(A)
            inner_prob_matrix = self.compute_inner_prob_matrix(A)
            output.append(torch.matmul(leafs_prob_matrix.permute(1, 0), self.param_dict['phi' + str(t)]))
            output_sum = output_sum + 0.1 * torch.squeeze(output[t])
            loss_Boosting = torch.exp(-torch.mul(output_sum, Y))
            residual = torch.mul(loss_Boosting, Y)
            cost_wr = torch.sum(torch.pow((residual - torch.squeeze(output[t])), 2))
            reg, V_now = self.compute_regularization(A, inner_prob_matrix, self.V_next_value[:, t])
            V_next.append(V_now)
            cost_sum = cost_sum + cost_wr + 1. * reg + 0.005 * torch.sum(torch.pow(self.param_dict['W' + str(t)], 2))
        V_next = torch.stack(V_next, 1)
        self.V_next_value = V_next.detach()

        return cost_sum


# function predict class by f1..f8 1d-array
def predict_1d(X, row_idx=0):
    # Preprocessing for xgb model
    d = dict()
    for i in range(8):
        d[f'f{i+1}'] = X[i]
    X_xgb = pd.DataFrame(d, index=range(1))

    # Predictions
    xgb_pred = (baseline.predict(xgb.DMatrix(X_xgb)) > 0.66).astype(int)[0]
    catboost_pred = catboost_model.predict(X)
    if X.min() != X.max():
        # Preprocessing for dbdt model
        y_ = np.array([0] * 2)
        y_ = y_ * 2 - 1
        X_ = np.array([list(X), list(X)])
        X_ = (X_ - X_.min()) / (X_.max() - X_.min())
        X_ = np.array(X_).astype(np.float32)
        y_ = np.array(y_).astype(np.float32)
        X_ = X_ * 2 - 1
        # Predict DBDT
        dbdt_pred = dbdt.compute_Boosting_output(torch.tensor(X_), torch.tensor(y_))
        dbdt_pred = np.array(torch.Tensor.tolist(dbdt_pred))
        dbdt_pred = (dbdt_pred > -0.711).astype(int)[0]
        return xgb_pred | catboost_pred | dbdt_pred
    else:
        print(f"# Prediction without DBDT model in row {row_idx}, features don't have to be the same")
        return xgb_pred | catboost_pred


# function predict class by f1..f8 nd-array
def predict_nd(X):
    n_x = X.shape[0]
    predict = list()
    for i in range(n_x):
        predict.append(predict_1d(X.iloc[i].values, row_idx=i))

    return np.array(predict)


if __name__ == '__main__':
    # Read data and output path
    if len(sys.argv) == 1:
        print('Write data:')
        X = list()
        for i in [f'f{i}' for i in range(1, 9)]:
            X.append(float(input(f'{i} = ')))
        X = np.array(X).astype(np.float32)
    else:
        input_path = sys.argv[1]
        try:
            X = pd.read_csv(input_path)
            print(f'Input data from file "{input_path}"')
            if X.shape[0] < 100:
                print(X.head(10))
            else:
                print(X)
            print('-' * 60)
        except:
            print(f'Error: File "{input_path}" is not exist.')
            exit(0)

        if len(sys.argv) == 3:
            output_path = sys.argv[2]
            if len(output_path.split('.')) != 2 or output_path.split('.')[1] != 'csv':
                output_path += '.csv'
        else:
            output_path = 'output.csv'
        print(f'Print results in file "{output_path}"')
    start_time = datetime.now()

    # Init models
    try:
        baseline = xgb.Booster()
        baseline.load_model('baseline.json')
    except:
        print('Error: "baseline.json" is not exist.')
        exit(0)
    try:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model('catboost')
    except:
        print('Error: "catboost" is not exist.')
        exit(0)
    try:
        dbdt = SDTs(8, use_cuda=False, random_seed=0)
        dbdt.params = torch.load('model_adam.pth', map_location=torch.device('cpu')).values()
    except:
        print('Error: "model_adam.pth" is not exist.')
        exit(0)

    # Prediction by predict function with len(sys.argv) argument and output the results:
    if len(sys.argv) == 1:
        pred = predict_1d(X)
        print(f'Class prediction: {pred}')
    else:
        pred = predict_nd(X)
        print(pred)
        pd.Series(pred).to_csv(output_path, index=0)
    print(f'--time: {(datetime.now() - start_time).total_seconds()} seconds')