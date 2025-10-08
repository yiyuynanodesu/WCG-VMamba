'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from torch.utils.data import DataLoader

from APIs.APIDataset import APIDataset
from APIs.APIEncode import api_encode
from APIs.APIMetric import api_metric

class Processor:

    def __init__(self, config) -> None:
        self.config = config
        pass

    def __call__(self, data, params,mode):
        return self.to_loader(data, params, mode)

    def encode(self, data, mode):
        return api_encode(data,self.config, mode)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data, mode):
        dataset_inputs = self.encode(data, mode)
        return APIDataset(*dataset_inputs)

    def to_loader(self, data, params, mode):
        dataset = self.to_dataset(data,mode)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn, drop_last=False)