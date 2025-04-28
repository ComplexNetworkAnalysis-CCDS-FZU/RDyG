import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.modules import MergeLayer
from DyGLib.utils.utils import NegativeEdgeSampler, get_neighbor_sampler
from src.model.batch_emd_updater import BatchEmbeddingUpdater
from src.payload.dataset import BaseDataset
from DyGLib.utils.DataLoader import Data as DyGData
from src.utils.dataloader import Data


class LinkPredictionModel(torch.nn.Module):
    def __init__(
        self, node_raw_features, edge_raw_features, init_neighbor_sampler
    ) -> None:
        super().__init__()

        self.dyg_model = DyGFormer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=init_neighbor_sampler,
            time_feat_dim=100,
            channel_embedding_dim=50,
            patch_size=1,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            max_input_sequence_length=32,
            device="cpu",
        )

        self.r_part_model = BatchEmbeddingUpdater(128, 128)

        self.link_predict = MergeLayer(
            128, input_dim2=128, hidden_dim=128, output_dim=1
        )

    def forward(self, dataset: BaseDataset, batchtifier: torch.utils.data.DataLoader):
        src_embedding = torch.rand(
            [dataset.node_map().node_number(), 128], requires_grad=False
        )
        dst_embedding = torch.rand(
            [dataset.node_map().node_number(), 128], requires_grad=False
        )

        neg_src_embedding = torch.rand(
            [dataset.node_map().node_number(), 128], requires_grad=False
        )
        neg_dst_embedding = torch.rand(
            [dataset.node_map().node_number(), 128], requires_grad=False
        )

        rescue_epochs = tqdm(enumerate(batchtifier))
        for num, data in rescue_epochs:
            data = Data(
                src_node_ids=list(data["src_node_ids"]),
                dst_node_ids=list(data["dst_node_ids"]),
                node_interactive_time=list(data["node_interactive_time"]),
                edge_ids=list(data["edge_ids"]),
                label=list(data["label"]),
            )
            data = DyGData(
                data.src_node_ids,
                data.dst_node_ids,
                data.node_interactive_time,
                data.edge_ids,
                data.label,
            )

            neighbor_sampler = get_neighbor_sampler(data, seed=12345)
            self.dyg_model.set_neighbor_sampler(neighbor_sampler)

            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                _,
            ) = (
                np.asarray(data.src_node_ids),
                np.asarray(data.dst_node_ids),
                np.asarray(data.node_interact_times),
                np.asarray(data.edge_ids),
            )
            neg_edge_sampler = NegativeEdgeSampler(
                src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids
            )

            _, batch_neg_dst_node_ids = neg_edge_sampler.sample(
                size=len(batch_src_node_ids)
            )
            batch_neg_src_node_ids = batch_src_node_ids

            # 正样本
            batch_src_node_embedding, batch_dsc_node_embedding = (
                self.dyg_model.compute_src_dst_node_temporal_embeddings(
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                )
            )

            src_embedding, dst_embedding = self.r_part_model.forward(
                torch.from_numpy(batch_src_node_ids),
                torch.from_numpy(batch_dst_node_ids),
                src_embedding.clone(),
                dst_embedding.clone(),
                batch_src_node_embedding,
                batch_dsc_node_embedding,
            )

            # 负样本
            batch_neg_src_node_embedding, batch_neg_dst_node_embedding = (
                self.dyg_model.compute_src_dst_node_temporal_embeddings(
                    batch_neg_src_node_ids,
                    batch_neg_dst_node_ids,
                    batch_node_interact_times,
                )
            )
            neg_src_embedding, neg_dst_embedding = self.r_part_model.forward(
                torch.from_numpy(batch_neg_src_node_ids),
                torch.from_numpy(batch_neg_dst_node_ids),
                neg_src_embedding.clone(),
                neg_dst_embedding.clone(),
                batch_neg_src_node_embedding,
                batch_neg_dst_node_embedding,
            )

        # 链路预测
        postiive_probabilities = (
            self.link_predict.forward(src_embedding, dst_embedding)
            .squeeze(dim=-1)
            .sigmoid()
        )
        negtive_probabilities = (
            self.link_predict(neg_src_embedding, neg_dst_embedding)
            .squeeze(dim=-1)
            .sigmoid()
        )

        return postiive_probabilities, negtive_probabilities
