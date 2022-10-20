import pdb
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm

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


'''
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

        #data_dict = nx.read_gpickle(self.data_path_state + str(0))
        with open(self.data_path_state + str(0), "rb") as fh:
          data_dict = nx.read_gpickle(fh)
        self.pipe_label_to_index_dict = pipe_label_to_index(data_dict['flow_rate'])

        self.get_dict()


    def get_dict(self):
        self.data_dict = {}
        for i in tqdm(self.file_ids):
            data_dict = nx.read_gpickle(self.data_path_state + str(i))

            flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=self.dtype)
            head = torch.tensor(data_dict['head'].values, dtype=self.dtype)
            pars = data_dict['leak']['pipe']

            self.data_dict['flow_rate'] = flow_rate
            self.data_dict['head'] = head
            self.data_dict['pars'] = pars

            
        nx.write_gpickle(self.data_dict, f'lol')


    def transform_state(self, data):
        return self.transformer.transform_state(data)

    def inverse_transform_state(self, data):
        return self.transformer.inverse_transform_state(data)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):

        idx = self.file_ids[idx]

        #with open(self.data_path_state + str(idx), "rb") as fh:
        #  data_dict = pickle.load(fh)

        #flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=self.dtype)
        #head = torch.tensor(data_dict['head'].values, dtype=self.dtype)
        #demand = torch.tensor(data_dict['demand'].values, dtype=self.dtype)

        flow_rate = self.data_dict['flow_rate']
        head = self.data_dict['head']

        data = torch.cat([flow_rate, head], dim=1)

        
        if self.transformer is not None:
            data = self.transform_state(data)

        if True:#'leak' in self.data_dict:
            #pars = torch.zeros([flow_rate.shape[0],], dtype=self.dtype)
            #pars[self.pipe_label_to_index_dict[data_dict['leak']['pipe']]] = 1

            #pars = torch.tensor(
            #    [self.pipe_label_to_index_dict[data_dict['leak']['pipe']]],
            #    dtype=torch.int32
            #)
            pars = self.data_dict['pars']
            

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

        t = torch.tensor(data_dict['head'].index.to_numpy()/60/60 % 24, dtype=torch.int32)
        pars = pars.repeat(t.shape[0])
        #pars = pars.unsqueeze(0).repeat(2, 1)

        
        
        pars = torch.stack([pars, t], dim=1)

        return data, pars#.unsqueeze(1)


'''

class NetworkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path,
            file_ids=range(10000),
            transformer=None,
            no_leak_classification=False,
            sensors=None,
            with_time=True
    ):

        self.no_leak_classification = no_leak_classification
        self.data_path_state = data_path
        self.file_ids = file_ids
        self.transformer = transformer
        self.with_time = with_time

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

        flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=self.dtype)
        head = torch.tensor(data_dict['head'].values, dtype=self.dtype)
        #demand = torch.tensor(data_dict['demand'].values, dtype=self.dtype)

        flow_rate = flow_rate[0:24]
        head = head[0:24]

        data = torch.cat([flow_rate, head], dim=1)

        
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

        t = torch.tensor(data_dict['head'].index.to_numpy()/60/60 % 24, dtype=torch.int32)
        t = t[0:24]
        pars = pars.repeat(t.shape[0])

        if self.with_time:
            pars = torch.stack([pars, t], dim=1)
        else:
            pars = pars.unsqueeze(1)

        return data, pars#.unsqueeze(1)
