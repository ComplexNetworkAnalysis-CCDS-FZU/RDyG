from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler, get_neighbor_sampler
from src.utils.batchtifier import BatchedDataset
from src.utils.batchtifier.global_event_batch import GlobalEventBatchtifier
from src.utils.dataloader import SplitEventStream
from torch.utils.data import DataLoader

# TODO： 解决模型初始化参数问题
node_raw_features = None
edge_raw_features = None
train_neighbor_sampler = None

# TODO: 数据集加载为EventStream
event_stream = None

# 划分数据集
train,test = SplitEventStream.spilt_event_stream_at(event_stream,int(len(event_stream)*0.9))

train_batchtify = GlobalEventBatchtifier(train,400)
test_batchtify = GlobalEventBatchtifier(test,400)

train_dataset = BatchedDataset(train_batchtify)
test_dataset = BatchedDataset(test_batchtify)

train_dataloader = DataLoader(train_dataset)
test_dataloader = DataLoader(test_dataset)


init_neighbor_sampler = NeighborSampler([])

model = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=init_neighbor_sampler,
                                         time_feat_dim=100, channel_embedding_dim=50, patch_size=1,
                                         num_layers=2, num_heads=2, dropout=0.1,
                                         max_input_sequence_length=32, device='cuda')


for epoch in range(100):
    for data in train_dataloader:
        data:Data
        model.train()
        
        neighbor_sampler = get_neighbor_sampler(data)
        model.set_neighbor_sampler(neighbor_sampler)
        
        batch_src_node_ids, batch_dst_node_ids,batch_node_interact_times,batch_edge_ids = data.src_node_ids,data.dst_node_ids,data.node_interact_times,data.edge_ids
        
        
        batch_src_node_embedding ,batch_dsc_node_embedding = model.compute_src_dst_node_temporal_embeddings(
            batch_src_node_ids,
            batch_dst_node_ids,
            batch_node_interact_times
        )
        
        # TODO: 接入后续模型部分
        
        # TODO: 损失函数与梯度下降
        