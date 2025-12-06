


import torch
import torch.nn as nn
import torch.nn.functional as F

class SDL(nn.Module):
    def __init__(self, dim, k, size):
        """
        Args:
            dim (int): Dimension of the feature vector.
            k (int): Number of classes or clusters.
            size (int): Size of the queue for storing features and targets.
        """
        super(SDL, self).__init__()
        self.dim = dim
        self.k = k
        self.size = size

        self.queue = torch.zeros(size, dim).cuda()  
        self.targets = torch.zeros(size).cuda()  
        self.ptr = 0 

    def forward(self, x, outputs):
      
        x_norm = F.normalize(x, dim=1)
        queue_norm = F.normalize(self.queue, dim=1)
        cosine_similarity = torch.mm(x_norm, queue_norm.T)
        return cosine_similarity

    @torch.no_grad()
    def update(self, feat, outputs):
       
        batch_size = feat.shape[0]

       
        if self.ptr + batch_size > self.size:
            overflow = (self.ptr + batch_size) - self.size
            self.queue[self.ptr:self.size] = feat[:batch_size - overflow]
            self.targets[self.ptr:self.size] = outputs[:batch_size - overflow]
            self.queue[:overflow] = feat[batch_size - overflow:]
            self.targets[:overflow] = outputs[batch_size - overflow:]
            self.ptr = overflow
        else:
            self.queue[self.ptr:self.ptr + batch_size] = feat
            self.targets[self.ptr:self.ptr + batch_size] = targets
            self.ptr = (self.ptr + batch_size) % self.size