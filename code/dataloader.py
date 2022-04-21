import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import parse

class MemLoader(Dataset):
    '''
    Memorization management
    Function: generate and update memorized data
    '''
    def __init__(self, config):
        self.path = f'../data/{config.dataset}'
        self.dataset = config.dataset
        self.history_len = config.history_len
        self.n_user = 0
        self.m_item = 0
        self.config = config

        print('Preparing memloader...')

        train_file = self.path + f'/{self.dataset}.train.rating'

        train_data = pd.read_csv(
            train_file,
            sep='\t', header=None, names=['user', 'item', 'noisy'],
            usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32}
        )
        if self.dataset == 'adressa':
            self.n_user = 212231
            self.m_item = 6596
        else:
            self.n_user = train_data['user'].max() + 1
            self.m_item = train_data['item'].max() + 1

        # record number of iteractions of each user
        self.user_pos_counts = pd.value_counts(train_data['user']).sort_index()

        self.trainUniqueUsers = np.array(range(self.n_user))
        self.trainUser = train_data['user'].values
        self.trainItem = train_data['item'].values
        self.traindataSize = len(self.trainItem)

        # memorization history matrix, 1 for memorized and 0 for non-memorized
        self.mem_dict = np.zeros((self.traindataSize, self.history_len), dtype=np.int8)
        # loop pointer that indicates current epoch, increment at the beginning of each epoch
        self.mem_dict_p = -1
        # map index from (u,i) to row position of memorization history matrix
        self.index_map = np.zeros((self.n_user, self.m_item), dtype=np.int32)
        self.index_map[:, :] = -1
        for ii in range(self.traindataSize):
            u = self.trainUser[ii]
            i = self.trainItem[ii]
            self.index_map[u][i] = ii

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    def updateMemDict(self, users, items):
        '''
        users and items: memorized pairs
        '''
        # increment pointer
        self.mem_dict_p += 1
        # loop pointer
        self.mem_dict_p %= self.history_len
        # initialize (clear) memorization record of current epoch
        self.mem_dict[:, self.mem_dict_p] = 0

        indexes = []
        for i in range(len(users)):
            index = self.index_map[users[i]][items[i]]
            if index != -1:
                indexes.append(index)
        self.mem_dict[indexes, self.mem_dict_p] = 1

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    def generate_clean_data(self):
        '''
        generate memorized data
        '''
        ismem_dict = np.sum(self.mem_dict, axis=1) >= self.history_len / 2
        mem_num = np.sum(ismem_dict)
        #print('Memory ratio:', mem_num / self.traindataSize)
        if mem_num > 0:
            indexes = np.argwhere(ismem_dict == True).reshape(1, -1)[0]
            clean_us = np.array(self.trainUser)[indexes]
            clean_is = np.array(self.trainItem)[indexes]
            clean_data = {'user': clean_us, 'item': clean_is}
            df = pd.DataFrame(clean_data)
            df.to_csv('./{}/clean_data_{}_{}.txt'.format(
                            self.dataset, self.config.model, self.config.lr),
                            header=False, index=False,sep='\t')
            return mem_num / self.traindataSize
        else:
            return False

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class Loader(Dataset):
    def __init__(self, config):
        self.path = f'../data/{config.dataset}'
        self.dataset = config.dataset

        print(f'loading [{self.path}]...')

        self.split = config.A_split
        self.folds = config.A_n_fold
        self.n_user = 0
        self.m_item = 0
        self.config = config

        train_file = self.path + f'/{self.dataset}.train.rating'
        test_file = self.path + f'/{self.dataset}.test.negative'
        valid_file = self.path + f'/{self.dataset}.valid.rating'
        trainItem, trainUser = [], []
        testUniqueUsers, testItem, testUser = [], [], []

        # loading training file
        with open(train_file, 'r') as f:
            line = f.readline()
            while line and line != '':
                arr = line.split('\t')
                u = int(arr[0])
                i = int(arr[1])
                self.m_item = max(self.m_item, i)
                self.n_user = max(self.n_user, u)
                trainUser.append(u)
                trainItem.append(i)
                line = f.readline()
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.array(list(set(trainUser)))
        self.traindataSize = len(trainUser)

        # loading validation file
        validUser, validItem, validUniqueusers = [], [], []
        with open(valid_file, 'r') as f:
            line = f.readline()
            while line and line != '':
                arr = line.split('\t')
                u = int(arr[0])
                i = int(arr[1])
                self.m_item = max(self.m_item, i)
                self.n_user = max(self.n_user, u)
                validUser.append(u)
                validItem.append(i)
                line = f.readline()
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)
        self.validUniqueUsers = np.array(list(set(validUser)))
        self.validdataSize = len(self.validItem)

        # loading test file
        with open(test_file, 'r') as f:
            line = f.readline()
            while line and line != '':
                arr = line.split('\t')
                if self.dataset == 'adressa':
                    u = eval(arr[0])[0]
                    i = eval(arr[0])[1]
                else:
                    u = int(arr[0])
                    i = int(arr[1])
                self.m_item = max(self.m_item, i)
                self.n_user = max(self.n_user, u)
                testUser.append(u)
                testItem.append(i)
                line = f.readline()

        self.m_item += 1
        self.n_user += 1
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.testUniqueUsers = np.array(list(set(testUser)))

        self.testdataSize = len(self.testItem)

        self.Graph = None

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        self.__validDict = self.__build_valid()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def evalDict(self):
        return self.__evalDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(parse.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(f'{self.path}/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(f'{self.path}/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(parse.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def __build_eval(self):
        eval_data = {}
        for i, item in enumerate(self.trainItem):
            user = self.trainUser[i]
            if eval_data.get(user):
                eval_data[user].append(item)
            else:
                eval_data[user] = [item]
        return eval_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class CleanLoader(Dataset):
    def __init__(self, config):
        self.path = f'{config.dataset}'
        self.split = config.A_split
        self.folds = config.A_n_fold
        self.n_user = 0
        self.m_item = 0
        self.config = config
        train_file = self.path + '/clean_data_{}_{}.txt'.format(config.model, config.lr)

        trainItem, trainUser = [], []

        with open(train_file, 'r') as f:
            line = f.readline()
            while line and line != '':
                arr = line.split('\t')
                u = int(arr[0])
                i = int(arr[1])
                self.m_item = max(self.m_item, i)
                self.n_user = max(self.n_user, u)
                # print(self.m_item)
                trainUser.append(u)
                trainItem.append(i)
                line = f.readline()
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.array(list(set(trainUser)))
        self.traindataSize = len(trainUser)

        self.m_item += 1
        self.n_user += 1

        self.Graph = None

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = None
        self.__validDict = None

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems