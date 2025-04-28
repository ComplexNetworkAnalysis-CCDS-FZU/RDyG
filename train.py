import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.modules import MLPClassifier, MergeLayer
from DyGLib.utils.DataLoader import Data as DyGData
from DyGLib.utils.utils import (
    NegativeEdgeSampler,
    NeighborSampler,
    get_neighbor_sampler,
)
from src.dataset.myket import MyketDataset
from src.model.batch_emd_updater import BatchEmbeddingUpdater
from src.model.link_predict_model import LinkPredictionModel
from src.utils.batchtifier import BatchedDataset
from src.utils.batchtifier.global_event_batch import GlobalEventBatchtifier
from src.utils.dataloader import Data, SplitEventStream
from torch.utils.data import DataLoader
from torch import autograd


# autograd.set_detect_anomaly(True)
# 数据集准备
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

# 模型准备
model = LinkPredictionModel(node_raw_features,edge_raw_features,init_neighbor_sampler)



# loss 与优化器
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1, weight_decay=0.0)

print(f"{dataset.node_map().node_number()}")

for epoch in range(100):

        postiive_probabilities,negtive_probabilities = model.forward(dataset,train_dataloader)

        # 损失函数与梯度下降

        predicts = torch.cat([postiive_probabilities, negtive_probabilities], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(postiive_probabilities),
                torch.zeros_like(negtive_probabilities),
            ],
            dim=0,
        )

        loss = loss_func(input=predicts, target=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, train_loss: {loss.item()}")
