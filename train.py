import numpy as np
import torch
from tqdm import tqdm
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.utils.DataLoader import Data as DyGData
from DyGLib.utils.utils import NeighborSampler, get_neighbor_sampler
from src.dataset.myket import MyketDataset
from src.model.batch_emd_updater import BatchEmbeddingUpdater
from src.utils.batchtifier import BatchedDataset
from src.utils.batchtifier.global_event_batch import GlobalEventBatchtifier
from src.utils.dataloader import Data, SplitEventStream
from torch.utils.data import DataLoader


dataset = MyketDataset()
node_raw_features = np.zeros([(dataset.node_map().node_number()), 128])
edge_raw_features = np.zeros([dataset.__len__(), 172])
train_neighbor_sampler = None


# 划分数据集
train, test = SplitEventStream.spilt_event_stream_at(
    dataset.event_stream(), int(len(dataset) * 0.9)
)

train_batchtify = GlobalEventBatchtifier(train, 400)
test_batchtify = GlobalEventBatchtifier(test, 400)

train_dataset = BatchedDataset(train_batchtify)
test_dataset = BatchedDataset(test_batchtify)

train_dataloader = DataLoader(train_dataset)
test_dataloader = DataLoader(test_dataset)


init_neighbor_sampler = NeighborSampler([])

model = DyGFormer(
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

r_part_model = BatchEmbeddingUpdater(128, 128)


for epoch in range(100):
    embedding = torch.rand([14096, 128])
    for data in tqdm(train_dataloader):
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
        model.train()

        neighbor_sampler = get_neighbor_sampler(data, seed=12345)
        model.set_neighbor_sampler(neighbor_sampler)

        (
            batch_src_node_ids,
            batch_dst_node_ids,
            batch_node_interact_times,
            batch_edge_ids,
        ) = (
            np.asarray(data.src_node_ids),
            np.asarray(data.dst_node_ids),
            np.asarray(data.node_interact_times),
            np.asarray(data.edge_ids),
        )

        batch_src_node_embedding, batch_dsc_node_embedding = (
            model.compute_src_dst_node_temporal_embeddings(
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
            )
        )

        embedding = r_part_model.forward(
            torch.from_numpy(batch_src_node_ids),
            torch.from_numpy(batch_dst_node_ids),
            embedding,
            batch_src_node_embedding,
            batch_dsc_node_embedding,
        )

        # TODO: 接入后续模型部分
        print(embedding)
        # TODO: 损失函数与梯度下降
        break
    break
