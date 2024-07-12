
import torch
from itertools import product,combinations
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature = 1) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self,embeddings,labels):
        #Normalize the embeddigs
        embeddings = F.normalize(embeddings, p=2, dim=1)

        positive_pairs = embeddings[labels == 0]
        negative_pairs = embeddings[labels == 1]
        num_pos_samples = positive_pairs.shape[0]
        num_neg_samples = negative_pairs.shape[0]

        
        dot_products_pos_pos = torch.ones([num_pos_samples,num_pos_samples],dtype=torch.float64,device=embeddings.device)
        dot_products_neg_neg = torch.ones([num_neg_samples,num_neg_samples],dtype=torch.float64,device=embeddings.device)
        dot_products_pos_neg = torch.ones([num_pos_samples,num_neg_samples],dtype=torch.float64,device=embeddings.device)

        for i,j in combinations(range(num_pos_samples),2):
            dot_products_pos_pos[i][j] = torch.exp(torch.dot(positive_pairs[i],positive_pairs[j])/self.temperature)
            dot_products_pos_pos[j][i] = dot_products_pos_pos[i][j]
        
        for i,j in combinations(range(num_neg_samples),2):
            dot_products_neg_neg[i][j] = torch.exp(torch.dot(negative_pairs[i],negative_pairs[j])/self.temperature)
            dot_products_neg_neg[j][i] = dot_products_neg_neg[i][j]

        for i,j in product(range(num_pos_samples),range(num_neg_samples)):
            dot_products_pos_neg[i][j] = torch.exp(torch.dot(positive_pairs[i],negative_pairs[j])/self.temperature)

        loss_pos = 0
        loss_neg = 0
        for i in range(num_pos_samples):
            denom = (torch.sum(dot_products_pos_pos[i,:i])+torch.sum(dot_products_pos_pos[i,i+1:]) + torch.sum(dot_products_pos_neg[i]))
            loss_pos += -(torch.sum(torch.log(dot_products_pos_pos[i,:i]/denom))+torch.sum(torch.log(dot_products_pos_pos[i,i+1:]/denom)))
        loss_pos = loss_pos/num_pos_samples

        for i in range(num_neg_samples):
            denom = (torch.sum(dot_products_neg_neg[i,:i])+torch.sum(dot_products_neg_neg[i,i+1:]) + torch.sum(dot_products_pos_neg[:,i]))
            loss_pos += -(torch.sum(torch.log(dot_products_neg_neg[i,:i]/denom))+torch.sum(torch.log(dot_products_neg_neg[i,i+1:]/denom)))
        loss_neg = loss_neg/num_neg_samples

        return loss_pos + loss_neg