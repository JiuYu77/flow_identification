# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader

def data_loader(Dataset, datasetPath, sampleLength:int, step:int, transform, clss=None, supervised=True,
                batchSize:int=64, shuffle=True, numWorkers=0):
    dataset = Dataset(datasetPath, sampleLength, step, transform, clss, supervised)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers)
    return dataloader

