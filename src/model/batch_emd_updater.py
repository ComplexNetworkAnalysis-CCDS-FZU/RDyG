from turtle import forward
import numpy
from torch import nn
import torch
from torch_geometric import nn as pyg_nn
DIM_ROW = 0
DIM_COLUMN = 1

class BatchEmbeddingUpdater(nn.Module):
    def __init__(self,num_nodes:int,dim_nig:int,dim_node:int) -> None:
        super().__init__()
        
        self.neighbor_layer = nn.Linear(dim_nig,dim_node)
        self.node_layer = nn.Linear(dim_node,dim_node)        
        
    
    def forward(self,node_ids:torch.Tensor,previous_embedding:torch.Tensor,batch_neighbor_embedding:torch.Tensor):
        """
        """
        
        previous_embedding = torch.index_select(previous_embedding,DIM_ROW,node_ids)
        
        neighbor_embedding_shift = self.neighbor_layer.forward(batch_neighbor_embedding)
        node_embedding = self.node_layer.forward(previous_embedding + neighbor_embedding_shift)
        return node_embedding
        