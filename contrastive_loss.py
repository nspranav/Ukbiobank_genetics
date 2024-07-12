import torch
from itertools import product
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature:float) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self,embeddings1,embeddings2):
        #Normalize the embeddigs
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        try:
            num_samples_batch = embeddings1.shape[0]
            dot_products_em1_em2 = torch.zeros([num_samples_batch,num_samples_batch],dtype=torch.float64)
            for i,j in product(range(num_samples_batch),repeat=2):
                dot_products_em1_em2[i][j]= torch.exp(torch.dot(embeddings1[i],embeddings2[j])/self.temperature)

            #creating clone for computing gene_smri embedding
            #dot_products_em2_em1 = torch.clone(dot_products_em1_em2)
            
            similarity_em1_em2 = 0
            similarity_em2_em1 = 0

            for i in range(num_samples_batch):
                similarity_em1_em2 += torch.log(dot_products_em1_em2[i][i]/(torch.sum(dot_products_em1_em2[i,:i])+torch.sum(dot_products_em1_em2[i,i+1:])))
                similarity_em2_em1 += torch.log(dot_products_em1_em2[i][i]/(torch.sum(dot_products_em1_em2[:i,i])+torch.sum(dot_products_em1_em2[i+1:,i])))

        except Exception as ex:
            print(ex)

        return -similarity_em1_em2, -similarity_em2_em1
