from torch_geometric.loader import DataLoader


class MultiPartLoaderV2(DataLoader):
    def __init__(self, dataset, num_parts, batch_sampler=None, **kwargs):
        self.num_parts = num_parts
        super(MultiPartLoaderV2, self).__init__(dataset, batch_sampler=batch_sampler, **kwargs)

    def __iter__(self):
        for indices in self.batch_sampler:
            multi_batch = [[] for _ in range(self.num_parts)]
            for idx in indices:
                for i in range(self.num_parts):
                    multi_batch[i].append(self.dataset[idx][i])
            for i in range(self.num_parts):
                multi_batch[i] = self.collate_fn(multi_batch[i])
            yield multi_batch


class MultiPartLoader(DataLoader):
    def __init__(self, dataset, batch_sampler):
        super(MultiPartLoader, self).__init__(dataset, batch_sampler=batch_sampler)

    def __iter__(self):
        for multi_indices in self.batch_sampler:
            multi_batch = []
            for batch_indices in multi_indices:
                batch = [self.dataset[i] for i in batch_indices]
                multi_batch.append(self.collate_fn(batch))
            yield multi_batch
