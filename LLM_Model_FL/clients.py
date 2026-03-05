import math

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import random
import matplotlib.pyplot as plt
import copy

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.round = 0
        self.count = 0

        # self.loss = 0

    def get_loss(self,localBatchSize, Net, lossFun, global_parameters):
        sum_loss = 0
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        with torch.no_grad():
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label, reduction='none')
                temp_loss = loss * loss
                sum_loss += temp_loss.sum()
                # loss = loss.mean()
                # loss.backward()
            sum_loss = math.sqrt(sum_loss /len(self.train_ds)) * len(self.train_ds)
        return sum_loss

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, global_parameters, opti):
        # self.loss = 0
        Net.load_state_dict(global_parameters, strict=True)

        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                # self.loss += loss*data.shape[0]
                loss.backward()
                opti.step()
                opti.zero_grad()
        
        # self.loss /= len(self.train_dl)
        self.local_parameters = copy.deepcopy(Net.state_dict())
        return Net.state_dict()



    def probing(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        loss_res = 0
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            loss_res += loss*data.shape[0]
            loss.backward()
            opti.step()
            opti.zero_grad()
        data_size = len(self.train_dl)
        util = math.sqrt(loss_res/data_size)*data_size
        train_time = self.get_time(localEpoch)

        return util, train_time


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, uniform, sigma):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.uniform = uniform
        self.sigma = sigma

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        '''Data and environment generation'''
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        if self.uniform == 'T':
            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            # np.random.seed(0)
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
        elif self.uniform == 'F':
            # random.seed(0)
            random_ratios = [random.random() for _ in range(self.num_of_clients)]
            total = sum(random_ratios)
            normalized_ratios = [r / total for r in random_ratios]

            sizes = [int(r * mnistDataSet.train_data_size) for r in normalized_ratios]
            data_start = 0
            sizes[-1] += mnistDataSet.train_data_size - sum(sizes)
            for i in range(self.num_of_clients):
                data_size = sizes[i]
                local_data = train_data[data_start:data_start+data_size]
                local_label = train_label[data_start:data_start+data_size]
                local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                data_start += data_size
        else:
            train_label = np.argmax(train_label, axis=1)
            client_idcs = self.dirichlet_split_noniid(train_label)
            for i in range(self.num_of_clients):
                local_data = train_data[client_idcs[i]]
                local_label = train_label[client_idcs[i]]
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone


    def dirichlet_split_noniid(self, train_labels):
        '''
        Partition the sample index set into n_clients subsets according to a Dirichlet distribution with parameter alpha.
        '''
        alpha = self.sigma
        n_classes = train_labels.max() + 1
        # np.random.seed(0)
        label_distribution = np.random.dirichlet([alpha] *  self.num_of_clients, n_classes)
        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]

        client_idcs = [[] for _ in range( self.num_of_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs,
                                              (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                      astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]


        empty_clients = [i for i, idcs in enumerate(client_idcs) if len(idcs) == 0]
        
        if empty_clients:
            for empty_client in empty_clients:

                largest_client = max(client_idcs, key=len)
                n_transfer = min(50, len(largest_client))  
                client_idcs[empty_client] = largest_client[:n_transfer]
                client_idcs[client_idcs.index(largest_client)] = largest_client[n_transfer:]

        return client_idcs


if __name__=="__main__":
    np_seed = 4
    np.random.seed(np_seed)
    client_num = 20
    di = 0.15
    di_str = str(di)[0]+str(di)[2:]
    MyClients = ClientsGroup('mnist', True, client_num, 1,'Di',di)


    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(10)]
    for i in range(client_num):
        for x, y in MyClients.clients_set['client{}'.format(i)].train_ds:
            label_distribution[y.numpy()].append(i)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, client_num + 1.5, 1),
             label=[i for i in range(10)], rwidth=0.5)
    plt.xticks(np.arange(client_num), ["Client %d" %
                                      c_id for c_id in range(client_num)], rotation=70)
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.show()
    plt.savefig('./figure/client{}_seed{}_'.format(client_num,np_seed)+di_str+'.png')
    print('finish')