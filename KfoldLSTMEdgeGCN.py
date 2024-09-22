import os
import torch
import torch.nn.functional as F
from torch.nn import Module, LSTM, Linear, BatchNorm1d, ReLU, \
    CrossEntropyLoss, NLLLoss, Sequential
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.nn import global_mean_pool, global_max_pool, TopKPooling
from torch_geometric.data import DataLoader
from timeSeriesCorrelation import ABIDEDataset
import matplotlib.pyplot as plt
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# use LSTM to extract temporal feature from ROI time series(node feature)
class timeSeriesLSTM(Module):
    def __init__(self, in_channels, hidden_channels, feature_channels, num_layers=1):
        super(timeSeriesLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.lstm = LSTM(in_channels, hidden_channels, num_layers, batch_first=True)
        self.fc = Linear(hidden_channels, feature_channels)

    def forward(self, x):

        h0 = torch.zeros(1, x.size(0), self.hidden_channels).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_channels).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# Sequentially connect the residual GCNConv and EdgeConv
class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = BatchNorm1d(out_channels)
        self.bn2 = BatchNorm1d(out_channels)
        self.relu = ReLU(inplace=True)

        # Use a linear layer to match dimensions for residual connection if necessary
        if in_channels != out_channels:
            self.residual = Linear(in_channels, out_channels)
        else:
            self.residual = None

    def forward(self, x, edge_index):
        identity = x
        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out = self.bn2(out)

        if self.residual is not None:
            identity = self.residual(identity)

        out = out + identity
        out = self.relu(out)

        return out

class EdgeGCNModel(Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeGCNModel, self).__init__()
        self.mlp1 = Sequential(Linear(2 * in_channels, hidden_channels), ReLU())
        self.mlp2 = Sequential(Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.ic1 = in_channels
        self.ic2 = hidden_channels
        self.ic3 = hidden_channels
        self.ic4 = hidden_channels * 16
        self.ic5 = hidden_channels * 4
        self.classes = 2

        self.conv1 = EdgeConv(self.mlp1, aggr='max')
        self.conv2 = EdgeConv(self.mlp2, aggr='max')
        self.conv3 = EdgeConv(self.mlp3, aggr='max')

        self.fc4 = Linear(hidden_channels, hidden_channels)
        self.fc5 = Linear(hidden_channels, hidden_channels)
        self.fc6 = Linear(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)

        self.linear = Linear(hidden_channels, 2)

        self.rb1 = ResidualBlock(self.ic1, self.ic2)
        self.pool1 = TopKPooling(self.ic3, ratio=0.8, multiplier=1, nonlinearity=torch.sigmoid)
        self.rb2 = ResidualBlock(self.ic3, self.ic4)
        self.pool2 = TopKPooling(self.ic4, ratio=0.8, multiplier=1, nonlinearity=torch.sigmoid)
        self.rb3 = ResidualBlock(self.ic4, self.ic5)

        self.fc1 = Linear((self.ic2 + self.ic4) * 2, self.ic2)
        self.bn1 = BatchNorm1d(self.ic2)
        self.fc2 = Linear(self.ic2, self.ic3)
        self.bn2 = BatchNorm1d(self.ic3)
        self.fc3 = Linear(self.ic3, self.classes)

    def forward(self, data):
        if data.edge_attr:
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        else:
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.weight

        x = self.conv1(x, edge_index)
        x = self.bn3(F.relu(self.fc4(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn4(F.relu(self.fc5(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)

        x = self.bn5(F.relu(self.fc6(x)))
        x = self.linear(x)

        x = self.rb1(x, edge_index)

        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.rb2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

# Concatenation of EdgeConv and residual GCNConv
# class EdgeGCNModel(Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(EdgeGCNModel, self).__init__()
#         self.mlp1 = Sequential(Linear(2 * in_channels, hidden_channels), ReLU())
#         self.mlp2 = Sequential(Linear(2 * hidden_channels, hidden_channels), ReLU())
#         self.mlp3 = Sequential(Linear(2 * hidden_channels, hidden_channels), ReLU())
#         self.ic1 = in_channels
#         self.ic2 = hidden_channels
#         self.ic3 = hidden_channels
#         self.ic4 = 512
#         self.ic5 = 256
#         self.classes = 2
#
#         self.edge1 = EdgeConv(self.mlp1, aggr='max')
#         self.edge2 = EdgeConv(self.mlp2, aggr='max')
#         self.edge3 = EdgeConv(self.mlp3, aggr='max')
#
#         self.fc4 = Linear(hidden_channels, hidden_channels)
#         self.fc5 = Linear(hidden_channels, hidden_channels)
#         self.fc6 = Linear(hidden_channels, hidden_channels)
#         self.bn3 = BatchNorm1d(hidden_channels)
#         self.bn4 = BatchNorm1d(hidden_channels)
#         self.bn5 = BatchNorm1d(hidden_channels)
#
#         self.gcn1 = GCNConv(self.ic1, self.ic2)
#         self.pool1 = TopKPooling(self.ic3, ratio=0.8, multiplier=1, nonlinearity=torch.sigmoid)
#         self.gcn2 = GCNConv(self.ic3, self.ic4)
#         self.pool2 = TopKPooling(self.ic4, ratio=0.8, multiplier=1, nonlinearity=torch.sigmoid)
#
#         self.fc1 = Linear((self.ic2 + self.ic4) * 2, self.ic2)
#         self.bn1 = BatchNorm1d(self.ic2)
#         self.fc2 = Linear(self.ic2, self.ic3)
#         self.bn2 = BatchNorm1d(self.ic3)
#         self.fc3 = Linear(self.ic3, self.ic4)
#         self.bn6 = BatchNorm1d(self.ic4)
#         self.fc7 = Linear(self.ic4, hidden_channels)
#         self.bn7 = BatchNorm1d(hidden_channels)
#
#         self.fc8 = Linear(2 * hidden_channels, self.ic5)
#         self.bn8 = BatchNorm1d(self.ic5)
#         self.fc9 = Linear(self.ic5, self.classes)
#
#     def forward(self, data):
#         if data.edge_attr is not None:
#             x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
#         else:
#             x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.weight
#
#         x_edge = self.edge1(x, edge_index)
#         x_edge = self.bn3(F.relu(self.fc4(x_edge)))
#         x_edge = self.edge2(x_edge, edge_index)
#         x_edge = self.bn4(F.relu(self.fc5(x_edge)))
#         x_edge = self.edge3(x_edge, edge_index)
#         x_edge = self.bn5(F.relu(self.fc6(x_edge)))
#         x_edge = F.dropout(x_edge, p=0.5, training=self.training)
#         x_edge = global_mean_pool(x_edge, batch)
#
#         x_gcn = self.gcn1(x, edge_index)
#         x_gcn, edge_index, edge_attr, batch, perm, score1 = self.pool1(x_gcn, edge_index, edge_attr, batch)
#         x1 = torch.cat([global_mean_pool(x_gcn, batch), global_max_pool(x_gcn, batch)], dim=1)
#         edge_attr = edge_attr.squeeze()
#         edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x_gcn.size(0))
#         x_gcn = self.gcn2(x_gcn, edge_index)
#         x_gcn, edge_index, edge_attr, batch, perm, score1 = self.pool2(x_gcn, edge_index, edge_attr, batch)
#         x2 = torch.cat([global_mean_pool(x_gcn, batch), global_max_pool(x_gcn, batch)], dim=1)
#
#         x_gcn = torch.cat([x1, x2], dim=1)
#
#         x_gcn = self.bn1(F.relu(self.fc1(x_gcn)))
#         x_gcn = self.bn2(F.relu(self.fc2(x_gcn)))
#
#         x_gcn = self.bn6(F.relu(self.fc3(x_gcn)))
#         x_gcn = F.dropout(x_gcn, p=0.5, training=self.training)
#         x_gcn = self.bn7(F.relu(self.fc7(x_gcn)))
#         x_gcn = F.dropout(x_gcn, p=0.5, training=self.training)
#         x = torch.cat([x_edge, x_gcn], dim=-1)
#
#         x = self.bn8(F.relu(self.fc8(x)))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.log_softmax(self.fc9(x), dim=-1)
#
#         return x
#
#     def augment_adj(self, edge_index, edge_weight, num_nodes):
#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
#                                                  num_nodes=num_nodes)
#         edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
#                                                   num_nodes)
#         edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
#                                          edge_weight, num_nodes, num_nodes,
#                                          num_nodes)
#         edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
#         return edge_index, edge_weight

# ---------------------Prepare the input for LSTM model---------------------------
def preprocess_data(data_dict):
    processed_data = {}
    for length, data_list in data_dict.items():
        x_data = []
        y_data = []
        for data in data_list:
            x_data.append(data.x)
            y_data.append(data.y)
        x_data = torch.stack(x_data)
        y_data = torch.tensor(y_data)
        processed_data[length] = (x_data, y_data)
    return processed_data

# --------------------------Convert float to decimal characters--------------------------
def floatToDecimalChars(num):
    num_str = str(num)
    decimal_point_index = num_str.find('.')
    if decimal_point_index != -1:
        decimal_part = num_str[decimal_point_index + 1:]
    else:
        decimal_part = ''
    return decimal_part

# ---------Train LSTM model, extract feature and reshape output as feature channels hyper parameter---------
def train_model(model, data, epochs=20, learning_rate=0.01):

    x_data, y_data = data
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = loss_fn(outputs[:, -1, :], y_data)
        loss.backward()
        optimizer.step()
        print(f'LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# use the output of test process to extract feature
def extract_features(model, x_data):
    model.eval()
    with torch.no_grad():
        features = model(x_data)
    return features

# --------------------------Train and test GCN model--------------------------
def train(model, loss_fn, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

def test(model, loss_fn, device, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(test_loader.dataset)

# Define the evaluation function
def evaluation_matrix(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = torch.argmax(model(batch), 1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1

# --------------------------Grid search hyperparameters--------------------------
class PyGModelWrapper:
    def __init__(self, kfnum, ROI, LSTM_hidden_channels, feature_channels, in_channels, fold, hidden_channels=32, lr=0.001,
                 batch_size=32, epochs=100, optimizer_name='Adam'):
        self.kfnum = kfnum,
        self.ROI = ROI,
        self.LSTM_hidden_channels = LSTM_hidden_channels,
        self.feature_channels = feature_channels,
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GCNmodel = EdgeGCNModel(in_channels, hidden_channels).to(self.device)
        self.optimizer = self._get_optimizer()
        self.batch_size = batch_size,
        self.epochs = epochs,
        self.loss_fn = CrossEntropyLoss()
        self.fold =fold

    def _get_optimizer(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.GCNmodel.parameters(), lr=self.lr)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.GCNmodel.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' not supported")

    def fit(self, train_loader, verify_loader, epochs=100):
        train_losses = []
        val_losses = []
        log_file_path = f"logs/filt_global_rois_{self.ROI[0]}/{self.kfnum}KF_LSTMEdgeGCNModel/{self.LSTM_hidden_channels}_{self.feature_channels}_" \
                        f"{self.hidden_channels}_{floatToDecimalChars(self.lr)}_" \
                        f"{self.batch_size}_{self.epochs}_{self.optimizer_name}.txt"
        log_path = f"logs/filt_global_rois_{self.ROI[0]}/{self.kfnum}kF_LSTMEdgeGCNModel/{self.LSTM_hidden_channels}_{self.feature_channels}_" \
                        f"{self.hidden_channels}_{floatToDecimalChars(self.lr)}_"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        running_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(log_file_path, 'a') as file:
            file.write(f'Running Time{running_timestamp}\n')
            file.write(
                f'Training with LSTM_hidden_channels={self.LSTM_hidden_channels}, '
                f'feature_channels={self.feature_channels}, fold={self.fold}'
                f'hidden_channels={self.hidden_channels}, lr={self.lr}, batch_size={self.batch_size}, '
                f'epochs={self.epochs}, optimizer_name={self.optimizer_name}\n')
        for epoch in range(epochs):
            train_loss = train(self.GCNmodel, self.loss_fn, self.device, train_loader, self.optimizer)
            # train_accuracy, train_precision, train_recall, train_f1 = evaluation_matrix(self.GCNmodel, self.device, train_loader)
            # val_loss = test(self.GCNmodel, self.loss_fn, self.device, verify_loader)
            val_accuracy, val_precision, val_recall, val_f1 = evaluation_matrix(self.GCNmodel, self.device, verify_loader)
            # with open(log_file_path, 'a') as file:
            #     file.write(f"epoch: {epoch}, train_loss: {train_loss:.6f}, train_acc: {train_accuracy:.6f}, "
            #                f"val_loss: {val_loss:.6f} val_acc: {val_accuracy:.6f}\n")
            # print(f"epoch: {epoch}, train_loss: {train_loss:.6f}, train_acc: {train_accuracy:.6f}, "
            #                f"val_loss: {val_loss:.6f} val_acc: {val_accuracy:.6f}\n")

            # with open(log_file_path, 'a') as file:
            #     file.write(f"epoch: {epoch}, train_loss: {train_loss:.6f}, "
            #                f"val_loss: {val_loss:.6f} val_acc: {val_accuracy:.6f}\n")
            # print(f"epoch: {epoch}, train_loss: {train_loss:.6f}, "
            #                f"val_loss: {val_loss:.6f} val_acc: {val_accuracy:.6f}\n")

            with open(log_file_path, 'a') as file:
                file.write(f"epoch: {epoch}, train_loss: {train_loss:.6f}, val_acc: {val_accuracy:.6f}\n")
            print(f"epoch: {epoch}, train_loss: {train_loss:.6f}, val_acc: {val_accuracy:.6f}\n")
            train_losses.append(train_loss)
            # val_losses.append(val_loss)
        # return train_losses, val_losses
        return train_losses

    def score(self, test_loader):

        log_file_path = f"logs/filt_global_rois_{self.ROI[0]}/{self.kfnum}KF_LSTMEdgeGCNModel/{self.LSTM_hidden_channels}_{self.feature_channels}_" \
                        f"{self.hidden_channels}_{floatToDecimalChars(self.lr)}_{self.batch_size}_{self.epochs}_" \
                        f"{self.optimizer_name}.txt"
        with open(log_file_path, 'a') as file:
            test_loss = test(self.GCNmodel, self.loss_fn, self.device, test_loader)
            test_accuracy, test_precision, test_recall, test_f1 = evaluation_matrix(self.GCNmodel, self.device, test_loader)
            file.write(f"test_loss: {test_loss:.6f}, test_acc: {test_accuracy:.6f}, test_pre: {test_precision:.6f}"
                       f", test_recall: {test_recall:.6f}, test_f1: {test_f1:.6f}\n")
            print(f"test_loss: {test_loss:.6f}, test_acc: {test_accuracy:.6f}, test_pre: {test_precision:.6f}"
                       f", test_recall: {test_recall:.6f}, test_f1: {test_f1:.6f}\n")
        return test_loss, test_accuracy, test_precision, test_recall, test_f1

# -------------Main function run the program------------


def main():
    best_acc = 0
    best_params = {}
    # adjust hyper parameters
    param_grid = {
        'ROIs': ['cc200', 'cc400'],
        'LSTM_hidden_channels': [32],
        'feature_channels': [2],
        'hidden_channels': [32],
        'lr': [0.01],
        'batch_size': [64],
        'epochs': [100],
        'optimizer_name': ['Adam']
    }
    kf_num = 5
    for ROI in param_grid['ROIs']:
    # ROI = 'cc200'
        for LSTM_hidden_channels in param_grid['LSTM_hidden_channels']:
            for feature_channels in param_grid['feature_channels']:
                print(f"t:{t}")
                # Initialize the dataset
                dataset = ABIDEDataset('filt_global_rois_' + ROI)
                data_dict = dataset._load_all_processed_files()
                model_dict = {}
                for length, data_list in data_dict.items():
                    model_dict[length] = timeSeriesLSTM(length, LSTM_hidden_channels,
                                                        feature_channels)

                processed_data = preprocess_data(data_dict)

                for length, model in model_dict.items():
                    print(f"Training model for sequence length {length}")
                    train_model(model, processed_data[length])

                for length, model in model_dict.items():
                    x_data, y_data = processed_data[length]
                    features = extract_features(model, x_data)
                    for i, data in enumerate(data_dict[length]):
                        data.x = features[i]

                all_data_list = []
                for length, data_list in data_dict.items():
                    all_data_list.extend(data_list)

                for hidden_channels in param_grid['hidden_channels']:
                    for lr in param_grid['lr']:
                        for batch_size in param_grid['batch_size']:
                            for epochs in param_grid['epochs']:
                                for optimizer_name in param_grid['optimizer_name']:

                                    kf = KFold(n_splits=kf_num, shuffle=True, random_state=123)
                                    test_all_losses = []
                                    test_all_accuracy = []
                                    test_all_precision = []
                                    test_all_recall = []
                                    test_all_f1 = []

                                    train_val_data, test_data = train_test_split(all_data_list, test_size=0.3,
                                                                                 shuffle=True,
                                                                                 random_state=42)
                                    fold_flag = 1
                                    for train_index, val_index in kf.split(train_val_data):
                                        train_subset = torch.utils.data.Subset(train_val_data, train_index)
                                        val_subset = torch.utils.data.Subset(train_val_data, val_index)
                                        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                                        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
                                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                                        model_wrapper = PyGModelWrapper(kfnum=kf_num, ROI=ROI, LSTM_hidden_channels=LSTM_hidden_channels,
                                                                        feature_channels=feature_channels,
                                                                        in_channels=feature_channels, fold=fold_flag,
                                                                        hidden_channels=hidden_channels, lr=lr,
                                                                        batch_size=batch_size, epochs=epochs,
                                                                        optimizer_name=optimizer_name)
                                        print(f"fold: {fold_flag}")
                                        # train_losses, val_losses = model_wrapper.fit(train_loader, val_loader, epochs)
                                        train_losses = model_wrapper.fit(train_loader, val_loader,
                                                                                     epochs)
                                        test_loss, test_accuracy, test_precision, test_recall, test_f1 = model_wrapper.score(test_loader)
                                        plt.clf()
                                        plt.plot(train_losses, color='red')
                                        # plt.plot(val_losses, color='green')
                                        plt.title(
                                            f"ROI={ROI} LSTM_hidden_channnels={LSTM_hidden_channels}, "
                                            f"feature_channels={feature_channels}, "
                                            f"hidden_channels={hidden_channels}\n"
                                            f"lr={lr}, batch_size={batch_size}, epochs={epochs}, "
                                            f"optimizer_name={optimizer_name}")
                                        plt.savefig(
                                            f"logs/filt_global_rois_{ROI}/{kf_num}KF_LSTMEdgeGCNModel/"
                                            f"{LSTM_hidden_channels}_{feature_channels}_"
                                            f"{hidden_channels}_{floatToDecimalChars(lr)}_{batch_size}_{epochs}_"
                                            f"{optimizer_name}_{fold_flag}.png")
                                        test_all_losses.append(test_loss)
                                        test_all_accuracy.append(test_accuracy)
                                        test_all_precision.append(test_precision)
                                        test_all_recall.append(test_recall)
                                        test_all_f1.append(test_f1)
                                        if fold_flag == kf_num:
                                            log_file_path = f"logs/test/records.txt"
                                            avg_loss = float(np.mean(test_all_losses))
                                            avg_accuracy = float(np.mean(test_all_accuracy))
                                            avg_precision = float(np.mean(test_all_precision))
                                            avg_recall = float(np.mean(test_all_recall))
                                            avg_f1 = float(np.mean(test_all_f1))
                                            running_timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                                                              time.localtime())
                                            with open(log_file_path, 'a') as file:
                                                file.write(f'Running Time{running_timestamp}\n')
                                                file.write(
                                                    f'ROI={ROI}, LSTM_hidden_channels={LSTM_hidden_channels}, '
                                                    f'feature_channels={feature_channels}'
                                                    f'hidden_channels={hidden_channels}, lr={lr}, batch_size={batch_size}, '
                                                    f'epochs={epochs}, optimizer_name={optimizer_name}\n')
                                                file.write(f"avg_test_loss: {avg_loss}, "
                                                           f"avg_test_accuracy: {avg_accuracy}, "
                                                           f"avg_test_precision: {avg_precision}, "
                                                           f"avg_test_recall: {avg_recall}, "
                                                           f"avg_test_f1: {avg_f1}\n")
                                            params = {
                                                        'ROI': ROI,
                                                        'LSTM_hidden_channels': LSTM_hidden_channels,
                                                        'feature_channels': feature_channels,
                                                        'hidden_channels': hidden_channels,
                                                        'lr': lr,
                                                        'batch_size': batch_size,
                                                        'epochs': epochs,
                                                        'optimizer_name': optimizer_name}
                                            print(f'Params: {params}')
                                            print(f"avg_test_loss: {avg_loss}, "
                                                  f"avg_test_accuracy: {avg_accuracy}, "
                                                  f"avg_test_precision: {avg_precision}, "
                                                  f"avg_test_recall: {avg_recall}, "
                                                  f"avg_test_f1: {avg_f1}")
                                            if avg_accuracy > best_acc:
                                                best_acc = avg_accuracy
                                                best_params = {
                                                    'ROI': ROI,
                                                    'LSTM_hidden_channels': LSTM_hidden_channels,
                                                    'feature_channels': feature_channels,
                                                    'hidden_channels': hidden_channels,
                                                    'lr': lr,
                                                    'batch_size': batch_size,
                                                    'epochs': epochs,
                                                    'optimizer_name': optimizer_name
                                                }

                                            print(f'Best Accuracy: {best_acc}')
                                            print(f'Best Params: {best_params}')
                                            continue
                                        else:
                                            fold_flag = fold_flag + 1

if __name__ == '__main__':
    main()