import pdb
import torch
import networkx as nx
import numpy as np

def pipe_label_to_index(flow_rate_df):

    pipe_labels = flow_rate_df.columns.to_list()
    pipe_index = [pipe_labels.index(pipe) for pipe in pipe_labels]
    pipe_label_to_index_dict = dict(zip(pipe_labels, pipe_index))

    return pipe_label_to_index_dict

def index_to_pipe_label(pipe_label_to_index_dict):

    pipe_index = list(pipe_label_to_index_dict.values())
    pipe_labels = list(pipe_label_to_index_dict.keys())
    index_to_pipe_label_dict = dict(zip(pipe_index, pipe_labels))

    return index_to_pipe_label_dict



class NetworkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path,
            file_ids=range(10000),
            transformer=None,
            no_leak_classification=False,
            sensors=None
    ):

        self.no_leak_classification = no_leak_classification
        self.data_path_state = data_path
        self.file_ids = file_ids
        self.transformer = transformer

        self.dtype = torch.get_default_dtype()

        self.sensors = sensors


        data_dict = nx.read_gpickle(self.data_path_state + str(0))
        self.pipe_label_to_index_dict = pipe_label_to_index(data_dict['flow_rate'])

    def transform_state(self, data):
        return self.transformer.transform_state(data)

    def inverse_transform_state(self, data):
        return self.transformer.inverse_transform_state(data)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):

        idx = self.file_ids[idx]

        data_dict = nx.read_gpickle(self.data_path_state + str(idx))

        flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=self.dtype)[0]
        head = torch.tensor(data_dict['head'].values, dtype=self.dtype)[0]
        demand = torch.tensor(data_dict['demand'].values, dtype=self.dtype)[0]

        data = torch.cat([flow_rate, head], dim=0)

        if self.transformer is not None:
            data = self.transform_state(data)

        if 'leak' in data_dict:
            #pars = torch.zeros([flow_rate.shape[0],], dtype=self.dtype)
            #pars[self.pipe_label_to_index_dict[data_dict['leak']['pipe']]] = 1

            pars = torch.tensor(
                    [self.pipe_label_to_index_dict[data_dict['leak']['pipe']]],
                    dtype=torch.int32
            )

        elif self.no_leak_classification:
            pars = torch.zeros([36,], dtype=self.dtype)
            pars[0] = torch.tensor(data_dict['leak']['demand'], dtype=self.dtype)
            pars[data_dict['leak']['pipe']] = 1
            #data = torch.cat([data, pars], dim=0)

        if self.sensors is not None:
            flow_rate_sensors = data[0:self.transformer.num_pipes][self.sensors['flow_rate_sensors']]
            head_sensors = data[-self.transformer.num_pipes:][self.sensors['head_sensors']]
            data = torch.cat([flow_rate_sensors, head_sensors], dim=0)

            if 'leak' in data_dict:
                #pars = torch.zeros([flow_rate.shape[0],], dtype=self.dtype)
                #pars[self.pipe_label_to_index_dict[data_dict['leak']['pipe']]] = 1

                pars = torch.tensor(
                        [self.pipe_label_to_index_dict[data_dict['leak']['pipe']]],
                        dtype=torch.int32
                )


        return data, pars


















