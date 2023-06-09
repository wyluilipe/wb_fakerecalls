from libauc.optimizers import PDSCA
from libauc.losses import CompositionalAUCLoss
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

# Data Preprocessing
torch.set_default_tensor_type(torch.DoubleTensor)

features = [f'f{i}' for i in range(1, 9)]
X, y = data[features], data['label']
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, stratify=y)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, train_size=0.5, stratify=y_valid)

x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_valid = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)

y_train = np.array(y_train).astype(np.float32)
y_valid = np.array(y_valid).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

y_train = y_train.reshape(-1)
y_valid = y_valid.reshape(-1)
y_test = y_test.reshape(-1)

y_train = y_train * 2 - 1
y_valid = y_valid * 2 - 1
y_test = y_test * 2 - 1

#SDTs class
class SDTs(nn.Module):
    def __init__(
            self,
            n_x,
            tree_levels=4,
            lmbd=0.1,
            T=100,
            imratio=None,
            margin=1,
            backend='ce',
            use_cuda=True):
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
        self.backend = 'ce'  # TODO:

        self.V_next_value = torch.zeros([self.inner_numb, self.T])

        self.param_dict = {}
        for i in range(T):
            self.param_dict['W' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(n_x, self.inner_numb)).cuda(), gain=1))
            self.param_dict['b' + str(i)] = nn.Parameter(
                torch.nn.init.constant_(Variable(torch.randn(1, self.inner_numb)).cuda(), 0))
            self.param_dict['phi' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(self.leafs_numb, 1), ).cuda(), gain=1))

        self.params = self.param_dict.values()

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]

    def node_probability(self, index_node, A):
        p = torch.ones(A.shape[0]).cuda()
        while index_node - 1 >= 0:
            father_index = int((index_node - 1) / 2)
            if (index_node - 1) % 2 == 0:
                p = p * (1.0 - A[:, father_index])
            else:
                p = p * (A[:, father_index])
            index_node = father_index
        return p

    def forward_propagation_Boosting(self, X, W, b):
        Z = torch.add(torch.matmul(X, W), b)
        A = torch.sigmoid(1 * Z)
        return A

    def compute_leafs_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.leafs_numb:
            ta.append(self.node_probability(self.leafs_numb - 1 + i, A))
            i = i + 1
        leafs_prob_matrix = torch.stack(ta, dim=0)
        return leafs_prob_matrix

    def compute_inner_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.inner_numb:
            ta.append(self.node_probability(i, A))
            i = i + 1
        inner_prob_matrix = torch.stack(ta, dim=0)
        return inner_prob_matrix

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

    def compute_Boosting_output(self, X, Y):
        output_sum = torch.full_like(Y, Y.mean())
        output = []

        for t in range(self.T):
            A = self.forward_propagation_Boosting(X, self.param_dict['W' + str(t)], self.param_dict['b' + str(t)])
            leafs_prob_matrix = self.compute_leafs_prob_matrix(A)
            inner_prob_matrix = self.compute_inner_prob_matrix(A)
            output.append(torch.matmul(leafs_prob_matrix.permute(1, 0), self.param_dict['phi' + str(t)]))
            output_sum = output_sum + 0.1 * torch.squeeze(output[t])
        return output_sum

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

    def forward(self, X, Y):
        y_pred = self.compute_Boosting_output(X, Y)
        y_true = Y
        if len(y_pred) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true) == 1:
            y_true = y_true.reshape(-1, 1)
        if self.backend == 'ce':
            self.backend = 'auc'

            return self.compute_cost_Boosting_wr(X, Y)
        else:
            self.backend = 'ce'
            if self.p is None:
                self.p = (y_true == 1).float().sum() / y_true.shape[0]
                y_pred = torch.sigmoid(y_pred)
                print(self.p, self.alpha, self.margin)
                self.L_AUC = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
                             self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
                             2 * self.alpha * (self.p * (1 - self.p) * self.margin + \
                                               torch.mean((self.p * y_pred * (0 == y_true).float() - (
                                                       1 - self.p) * y_pred * (1 == y_true).float()))) - \
                             self.p * (1 - self.p) * self.alpha ** 2

            return self.L_AUC

    def compute_accuracy_Boosting(self, X, Y):
        #         print('testing phi', self.param_dict['phi' + str(0)])
        output_sum = torch.full_like(Y, Y.mean())
        Y = Y.cuda()
        output = []
        for t in range(self.T):
            A = self.forward_propagation_Boosting(X, self.param_dict['W' + str(t)], self.param_dict['b' + str(t)])
            leafs_prob_matrix = self.compute_leafs_prob_matrix(A)
            output.append(torch.matmul(leafs_prob_matrix.permute(1, 0), self.param_dict['phi' + str(t)]))
            output_sum = output_sum + 0.1 * torch.squeeze(output[t])
        predictions = (torch.tanh(output_sum) > 0).type(torch.FloatTensor) - (torch.tanh(output_sum) < 0).type(
            torch.FloatTensor)
        predictions = predictions.type(torch.DoubleTensor).cuda()

        return predictions

if __name__ == '__main__':
    batch_size = 512
    epochs = 200
    t = 40
    #     imratio = 0.95
    margin = 2
    lr = 0.01
    gamma = 500
    weight_decay = 1e-4
    beta1 = 0.999  # try different values: e.g., [0.999, 0.99, 0.9]
    beta2 = 0.999  # try different values: e.g., [0.999, 0.99, 0.9]
    train_data = torch.tensor(x_train).cuda()
    train_labels = torch.tensor(y_train).cuda()
    eval_data = torch.tensor(x_valid).cuda()
    eval_labels = torch.tensor(y_valid).cuda()

    train_dataset = Data.TensorDataset(train_data, train_labels)
    eval_dataset = Data.TensorDataset(eval_data, eval_labels)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    test_loader = Data.DataLoader(
        eval_dataset,
        batch_size=batch_size
    )
    (m, n_x) = train_data.shape
    costs = []
    minibatches = int(m / batch_size)
    sdts = SDTs(n_x)
    sdts = sdts.cuda()
    alpha = 0.05

    def custom_loss_fn(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))

    optimizer = PDSCA(sdts,
                      a=sdts.a,
                      b=sdts.b,
                      alpha=None,
                      lr=lr,
                      loss_fn=CompositionalAUCLoss(),
                      beta1=beta1,
                      beta2=beta2,
                      gamma=gamma,
                      margin=margin,
                      weight_decay=weight_decay)

    optimizer = torch.optim.Adam(sdts.params, lr=0.1)

    test_auc_max = 0
    print('-' * 30)
    best_testing_auc = 0.0
    torch.set_grad_enabled(True)
    for epoch in range(epochs):
        # Training
        sdts.train()
        epoch_cost = 0
        if epoch == int(0.5 * epochs) or epoch == int(0.75 * epochs):
            optimizer.update_regularizer(decay_factor=10)
        train_pred = []
        train_true = []
        for batch_idx, (data, target) in enumerate(train_loader):
            torch.cuda.empty_cache()
            data = data
            target = target.view(-1)

            optimizer.zero_grad()

            y_pred = sdts.compute_Boosting_output(data, target)
            loss = sdts.forward(data, target)

            loss.backward(retain_graph=True)
            optimizer.step()

            train_pred.append(y_pred.cpu().detach().numpy())
            train_true.append(target.cpu().detach().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)
        # Evaluating
        sdts.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        test_pred = sdts.compute_Boosting_output(eval_data, eval_labels)
        val_auc = roc_auc_score(eval_labels.cpu().detach().numpy(), test_pred.cpu().detach().numpy())
        predictions = (torch.tanh(test_pred) > 0).type(torch.FloatTensor) - (torch.tanh(test_pred) < 0).type(
            torch.FloatTensor)
        predictions = predictions.type(torch.DoubleTensor).cuda()

        if test_auc_max < val_auc:
            test_auc_max = val_auc
            print(
                classification_report(predictions.cpu().detach().numpy(), eval_labels.cpu().detach().numpy(), digits=8))

        # print results
        print("epoch: {}, train_auc:{:4f}, test_auc:{:4f}, test_auc_max:{:4f}".format(epoch, train_auc, val_auc,
                                                                                      test_auc_max)) #optimizer.lr))
        sdts.train()
        torch.set_grad_enabled(True)
