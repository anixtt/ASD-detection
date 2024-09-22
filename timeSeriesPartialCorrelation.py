import os
import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx

import networkx as nx
from networkx import from_numpy_array

# InMemoryDataset:
# Dataset base class for creating graph datasets which
# easily fit into CPU memory.
# DevDataset inherits InMemoryDataset aim to prepare gnn data
class ABIDEDataset(InMemoryDataset):
    # root: Root directory where the dataset should be saved
    # transform: The data object will be transformed before every access
    # pre_transform: The data object will be transformed before being saved to disk
    def __init__(self, root, transform=None, pre_transform=None, neighbors_ratio=0.1):
        self.neighbors_ratio = neighbors_ratio
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    # The name of the files in the self.processed_dir folder
    # that must be present in order to skip processing.
    @property
    def processed_file_names(self):
        time_series_length = [195, 205, 77, 175, 145, 295, 235, 231, 315, 245, 151, 123, 176, 233, 115]
        return [f'data_{length}.pt' for length in time_series_length]
        # return ['data.pt']
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """

        dataset_dict = {}
        labels_dict = pd.read_csv(labels_file, header=None, index_col=0).squeeze().to_dict()

        # Paths of connectivity matrices
        pcorr_path_list = sorted(os.listdir(pcorr_matrices_dir), key=lambda x: int(x[-21:-15]))
        for i in range(0, len(pcorr_path_list)):
            pcorr_matrix_path = os.path.join(pcorr_matrices_dir, pcorr_path_list[i])
            time_series_path = os.path.join(time_series_dir, pcorr_path_list[i].split('.')[0]+'.1D')

            pcorr_matrix_np = np.loadtxt(pcorr_matrix_path, delimiter=',')
            time_series_np = np.loadtxt(time_series_path)
            if time_series_np.shape[0] not in dataset_dict:
                dataset_dict[time_series_np.shape[0]] = []

            node_features = torch.tensor(time_series_np.T, dtype=torch.float)
            n_rois = pcorr_matrix_np.shape[0]
            index = np.abs(pcorr_matrix_np).argsort(axis=1)
            # Take only the top k correlates to reduce number of edges
            for j in range(n_rois):
                for k in range(n_rois - int(self.neighbors_ratio * n_rois)):    #self.neighbors_ratio=0.1 n_rois=200
                    pcorr_matrix_np[j, index[j, k]] = 0

            pcorr_matrix_nx = from_numpy_array(pcorr_matrix_np)
            pcorr_matrix_data = from_networkx(pcorr_matrix_nx)
            pcorr_matrix_data.x = node_features
            label_key = pcorr_path_list[i].split('.')[0]
            pcorr_matrix_data.y = torch.from_numpy(np.array(labels_dict[label_key])).type(torch.LongTensor)
            # data = Data(x=node_features, edge_index=pcorr_matrix_data.edge_index, edge_attr=pcorr_matrix_data.weight, y=y)

            # dataset_dict[time_series_np.shape[0]].append(data)
            dataset_dict[time_series_np.shape[0]].append(pcorr_matrix_data)
        # Ensure the processed directory exists
        processed_dir = os.path.dirname(self.processed_paths[0])
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        # Save each length-specific dataset
        for length, data_list in dataset_dict.items():
            file_name = os.path.join(processed_dir, f'data_{length}.pt')  # Correct path construction
            torch.save(data_list, file_name)
            print(f"Saved dataset with length {length} to {file_name}")

    def _load_all_processed_files(self):
        data_dict = {}
        processed_dir = os.path.dirname(self.processed_paths[0])
        time_series_length = [195, 205, 77, 175, 145, 295, 235, 231, 315, 245, 151, 123, 176, 233, 115]
        for length in time_series_length:
            file_name = os.path.join(processed_dir, f'data_{length}.pt')
            data = torch.load(file_name)
            data_dict[length] = data
        return data_dict

if __name__ == '__main__':
    rois = ['rois_cc200', 'rois_cc400']
    # define the path of each variables
    dataset_path = '../dataset'
    for roi in rois:
        corr_matrices_dir = f'{dataset_path}/FCmatrics/dparsf/filt_global/' + roi + '/corr_matrices'
        pcorr_matrices_dir = f'{dataset_path}/FCmatrics/dparsf/filt_global/' + roi + '/pcorr_matrices'
        time_series_dir = f'{dataset_path}/All/dparsf/filt_global/' + roi
        labels_file = f'{dataset_path}/FCmatrics/dparsf/filt_global/' + roi + '/labels.csv'
        dataset = ABIDEDataset('filt_global_rois_' + roi)
    print("Graph datasets have been built and saved.")

    # dataset = ABIDEDataset('filt_global_rois_cc200_702')
    # all_data_dict = dataset._load_all_processed_files()
    # print(f"Loaded {len(all_data_dict)} datasets")
    # for length, data in all_data_dict.items():
    #     print(f"Length: {length}, Number of samples: {len(data)}")