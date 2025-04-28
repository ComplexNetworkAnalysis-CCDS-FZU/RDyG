from torch import nn
import torch

DIM_ROW = 0
DIM_COLUMN = 1


class BatchEmbeddingUpdater(nn.Module):
    def __init__(self, dim_nig: int, dim_node: int, dim_hidden: int = 256) -> None:
        super().__init__()

        self.src_neighbor_layer = nn.Linear(dim_nig, dim_hidden)
        self.src_node_resize_layer = nn.Linear(dim_node, dim_hidden)
        self.src_node_layer = nn.Linear(dim_hidden * 2, dim_node)

        self.dst_neighbor_layer = nn.Linear(dim_nig, dim_hidden)
        self.dst_node_resize_layer = nn.Linear(dim_node, dim_hidden)
        self.dst_node_layer = nn.Linear(dim_hidden * 2, dim_node)

    def forward(
        self,
        src_node_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        src_previous_embedding: torch.Tensor,
        dst_previous_embedding: torch.Tensor,
        batch_src_neighbor_embedding: torch.Tensor,
        batch_dst_neighbor_embedding: torch.Tensor,
    ):
        """ """

        selected_src_previous_embedding = self.src_node_resize_layer.forward(
            torch.index_select(src_previous_embedding, DIM_ROW, src_node_ids)
        )
        selected_dst_previous_embedding = self.dst_node_resize_layer.forward(
            torch.index_select(dst_previous_embedding, DIM_ROW, dst_node_ids)
        )

        src_neighbor_embedding_shift = self.src_neighbor_layer.forward(
            batch_src_neighbor_embedding
        )
        dst_neighbor_embedding_shift = self.dst_neighbor_layer.forward(
            batch_dst_neighbor_embedding
        )

        src_node_embedding = self.src_node_layer.forward(
            torch.cat(
                [selected_src_previous_embedding, src_neighbor_embedding_shift],
                DIM_COLUMN,
            )
        )

        dst_node_embedding = self.dst_node_layer.forward(
            torch.cat(
                [selected_dst_previous_embedding, dst_neighbor_embedding_shift],
                DIM_COLUMN,
            )
        )

        # 切断前一次留下梯度
        updated_src_embedding = src_previous_embedding.clone().detach()
        updated_src_embedding[src_node_ids] = src_node_embedding

        updated_dst_embedding = dst_previous_embedding.clone().detach()
        updated_dst_embedding[dst_node_ids] = dst_node_embedding

        return updated_src_embedding, updated_dst_embedding
