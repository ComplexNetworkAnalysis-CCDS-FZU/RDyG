from torch import nn
import torch

DIM_ROW = 0
DIM_COLUMN = 1


class BatchEmbeddingUpdater(nn.Module):
    def __init__(self, dim_nig: int, dim_node: int) -> None:
        super().__init__()

        self.neighbor_layer = nn.Linear(dim_nig, dim_node)
        self.node_layer = nn.Linear(dim_node, dim_node)

    def forward(
        self,
        src_node_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        previous_embedding: torch.Tensor,
        batch_src_neighbor_embedding: torch.Tensor,
        batch_dst_neighbor_embedding: torch.Tensor,
    ):
        """ """

        src_previous_embedding = torch.index_select(
            previous_embedding, DIM_ROW, src_node_ids
        )
        dst_previous_embedding = torch.index_select(
            previous_embedding, DIM_ROW, dst_node_ids
        )

        src_neighbor_embedding_shift = self.neighbor_layer.forward(
            batch_src_neighbor_embedding
        )
        dst_neighbor_embedding_shift = self.neighbor_layer.forward(
            batch_dst_neighbor_embedding
        )

        src_node_embedding = (
            self.node_layer.forward(
                src_previous_embedding + src_neighbor_embedding_shift
            )
            + src_previous_embedding
        )

        dst_node_embedding = (
            self.node_layer.forward(
                dst_previous_embedding + dst_neighbor_embedding_shift
            )
            + dst_previous_embedding
        )

        # 切断前一次留下梯度
        updated_embedding = previous_embedding.clone().detach()
        updated_embedding[src_node_ids] = src_node_embedding
        updated_embedding[dst_node_ids] = dst_node_embedding

        return updated_embedding
