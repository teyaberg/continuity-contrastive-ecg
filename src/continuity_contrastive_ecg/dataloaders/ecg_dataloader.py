import torch.utils.data

from continuity_contrastive_ecg.utils.module import Module


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_loader(dataset, cfg, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


class DatasetLoaders(Module):
    def __init__(self, cfg):
        self.cfg = cfg

        train_dataset = cfg.train_dataset
        eval_train_dataset = cfg.eval_train_dataset
        eval_test_dataset = cfg.eval_test_dataset
        self.train_loader = get_loader(train_dataset, self.cfg, shuffle=True)
        self.eval_train_loader = get_loader(eval_train_dataset, self.cfg, shuffle=False)
        self.eval_test_loader = get_loader(eval_test_dataset, self.cfg, shuffle=False)
