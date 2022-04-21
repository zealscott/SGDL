import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class Scheduler(nn.Module):
    def __init__(self, N):
        super(Scheduler, self).__init__()
        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 40
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, loss, input):
        grad_output, (hn, cn) = self.grad_lstm(input.reshape(1, len(input), -1))
        grad_output = grad_output.sum(0)

        loss_output, (hn, cn) = self.loss_lstm(loss.reshape(1, len(loss), 1))
        loss_output = loss_output.sum(0)

        x = torch.cat((grad_output, loss_output), dim=1)

        z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        p = prob.detach().cpu().numpy()
        if len(np.where(p > 0)[0]) < size:
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)), p=p / np.sum(p), size=size,
                                       replace=replace)
            actions = [torch.tensor(x).cuda() for x in actions]
        return torch.LongTensor(actions)

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps).cuda()

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = torch.eq(y, torch.max(y, 1, keepdim=True).values).long().cuda()
            y = (y_hard - y).detach() + y
            #y = torch.nonzero(y)[:, 1]
        return y