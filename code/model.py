from parse import args
import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src

                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad

                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            # if not same_var:
            #     param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaEmbed(MetaModule):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        ignore = nn.Embedding(dim_1, dim_2)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', None)

    def forward(self, index):
        return self.weight[index]

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class LightGCN(MetaModule):
    def __init__(self, dataset):
        super(LightGCN, self).__init__()
        self.config = args
        self.dataset = dataset

        self.__init_weight()
        self.store_params()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config.latent_dim_rec
        self.n_layers = self.config.lightGCN_n_layers
        self.keep_prob = self.config.keep_prob
        self.A_split = self.config.A_split
        self.embedding_user = MetaEmbed(self.num_users,self.latent_dim)
        self.embedding_item = MetaEmbed(self.num_items,self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        # [num_users, emb_dim]
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # [M + N, emb_dim]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        # [M + N, M + N] adjacency matrix
        g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # [M + N, emb_dim] E(k+1) = D^(-1/2)AD^(1/2)E(k)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # [M + N, num_layer, emb_dim]
        embs = torch.stack(embs, dim=1)

        # mean of all layers
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, users, pos, neg, reduce=True):
        # [batch_size, emb_dim]
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        # [batch_size, emb_dim]
        pos_scores = torch.mul(users_emb, pos_emb)
        # [batch_size, ]
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        if reduce:
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        else:
            loss = torch.nn.functional.softplus(neg_scores - pos_scores)
        return loss, reg_loss

    def loss_gumbel(self, users, pos, neg, reduce=True):
        # [batch_size, emb_dim]
        users_emb = users
        pos_emb = pos
        neg_emb = neg
        # [batch_size, emb_dim]
        pos_scores = torch.mul(users_emb, pos_emb)
        # [batch_size, ]
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        if reduce:
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        else:
            loss = torch.nn.functional.softplus(neg_scores - pos_scores)
        return loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def store_params(self):
        self.keep_weight = deepcopy(self.state_dict())
        self.fast_weights = OrderedDict()
        self.weight_names = list(self.keep_weight.keys())

class LTW(MetaModule):
    '''
    learning-to-weight module
    input:loss
    output:weight of each (u,i) pair
    '''
    def __init__(self, input, hidden1, output):
        super(LTW, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)