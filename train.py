import os
import time
import uuid
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from happy_config import ConfigLoader
from config import Config

loader = ConfigLoader(model=Config, config='/params/params_dss.json')
config = loader()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

import torch
print(f"Visible GPU index: {config.gpus}")
print("Cuda support:", torch.cuda.is_available(), ":", torch.cuda.device_count(), "devices")

import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from dataclasses import asdict
from torch_geometric import seed_everything
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader

from data.ee import EEV2 as EE
from data.bde import BDEV2 as BDE
from data.drugs_partial import DrugsV2 as Drugs
from data.kraken import KrakenV2 as Kraken

from loaders.multipart import MultiPartLoaderV2
from loaders.samplers import DistributedEnsembleSampler
from utils.checkpoint import load_checkpoint
from utils.early_stopping import EarlyStopping

from models.dss.painn_dss import PaiNN
from models.dss.clofnet_v2_dss import ClofNet
from models.dss.equiformer_dss import Equiformer
from models.dss.visnet_dss import ViSNet_DSS
from models.dss.topology import GIN
from models.spice import DSSNetV2

from utils.optim import get_optimizer, get_scheduler


def train(model, loader, optimizer, rank, epoch, batch_size, z_beta):
    model.train()

    total_loss = torch.zeros(2).to(rank)
    for i, data in enumerate(loader):

        optimizer.zero_grad()
        num_molecules = data[0].y.size(0)
        out, z_loss, feature = model(data, epoch, batch_size, loss_expected=None)

        loss = F.mse_loss(out, data[0].y) + z_beta * torch.mean(z_loss)

        loss.backward()
        optimizer.step()

        total_loss[0] += float(loss) * num_molecules
        total_loss[1] += num_molecules

    dist.barrier()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    loss = float(total_loss[0] / total_loss[1])
    return loss


def evaluate(model, loader, std, epoch, batch_size):
    model.eval()
    error = 0
    num_molecules = 0

    for data in loader:
        with torch.no_grad():
            out, z_loss, feature = model.module(data, epoch, loss_expected=None, batch_size=batch_size)

        error += ((out - data[0].y) * std).abs().sum().item()
        num_molecules += data[0].y.size(0)
    return error / num_molecules


def run(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    print(config.learning_rate)

    if config.dataset == 'Drugs':
        dataset = Drugs('datasets/Drugs', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'Kraken':
        dataset = Kraken('datasets/Kraken', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'BDE':
        dataset = BDE('datasets/BDE', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'EE':
        dataset = EE('datasets/EE', max_num_conformers=config.max_num_conformers)

    max_atomic_num = 100
    if config.modeldss.conf_encoder == 'PaiNN':
        conf_model_factory = lambda: PaiNN(max_atomic_num=max_atomic_num, **asdict(config.modeldss.painn))
    elif config.modeldss.conf_encoder == 'ClofNet':
        conf_model_factory = lambda: ClofNet(max_atomic_num=max_atomic_num, **asdict(config.modeldss.clofnet))
    elif config.modeldss.conf_encoder == 'Equiformer':
        conf_model_factory = lambda: Equiformer(
            max_atomic_num=max_atomic_num, **asdict(config.modeldss.equiformer))
    elif config.modeldss.conf_encoder == 'ViSNet':
        conf_model_factory = lambda: ViSNet_DSS(**asdict(config.modeldss.visnet))

    if config.modeldss.topo_encoder == 'GIN':
        topo_model_factory = lambda: GIN(hidden_dim=config.hidden_dim, output_dim=128, num_layers=6)

    seed_everything(config.seed)
    model = DSSNetV2(
        hidden_dim=config.hidden_dim, out_dim=1,
        conf_model_factory=conf_model_factory, topo_model_factory=topo_model_factory,
        num_experts=config.num_experts, num_activated=config.num_activated,
        num_parts=dataset.num_parts, device=f'cuda:{rank}',
        gig=config.gig, ad=config.ad, sag=config.sag, upc=config.upc, upcycling_epochs=config.upcycling_epochs,
        gumbel_tau=config.gumbel_tau).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    target_id = dataset.descriptors.index(config.target)
    dataset.y = dataset.y[:, target_id]

    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = ((dataset.y - mean) / std).to(rank)
    print(f'Dataset length: {len(dataset.y)}')
    mean = mean.to(rank)
    std = std.to(rank)

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [config.train_ratio, config.valid_ratio, 1 - config.train_ratio - config.valid_ratio])

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DistributedEnsembleSampler(
            dataset=train_dataset, num_replicas=world_size, rank=rank,
            batch_size=config.batch_size, batch_graph_size=config.batch_graph_size, seed=config.seed))
    if rank == 0:
        valid_loader = DataLoader(valid_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config, train_dataset, world_size)

    start_epoch = 0
    if config.checkpoint is None:
        checkpoint_path = (
            f'checkpoints/'
            f'{config.dataset}_{config.target}_'
            f'{config.modeldss.conf_encoder}_{config.modeldss.topo_encoder}_'
            f'{uuid.uuid4()}.pt')
        if rank == 0:
            print(f'Saving checkpoint to: {checkpoint_path}')
    else:
        checkpoint_path = config.checkpoint
    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer=optimizer,
                                                                   scheduler=scheduler)
        print(f'Loaded checkpoint: {checkpoint_path} at epoch {start_epoch} on rank {rank}')
        dist.barrier()
    if rank == 0:
        early_stopping = EarlyStopping(patience=config.patience, path=checkpoint_path)
        wandb.login()
        wandb.init(project=config.wandb_project, config=asdict(config))
        wandb.define_metric('epoch')
        wandb.define_metric('train_loss', step_metric='epoch')
        wandb.define_metric('valid_error', step_metric='epoch')
        wandb.define_metric('test_error', step_metric='epoch')
        print(f'Checkpoint path: {checkpoint_path}')
        print(f"ad={config.ad}, sag={config.sag}, gig={config.gig}, upc={config.upc}")
    else:
        early_stopping = None
    dist.barrier()

    for epoch in range(start_epoch, config.num_epochs):
        train_loader.batch_sampler.set_epoch(epoch)

        loss = train(model, train_loader, optimizer, rank, epoch, config.batch_size, config.z_beta)

        if scheduler is not None:
            scheduler.step(loss)
        print(f'Rank: {rank}, Epoch: {epoch}/{config.num_epochs}, Loss: {loss:.5f}')

        if rank == 0:
            valid_error = evaluate(model, valid_loader, std, epoch, config.batch_size)
            last_ckpt_path = early_stopping(valid_error, model, optimizer, scheduler, epoch)
            if early_stopping.counter == 0:
                test_error = evaluate(model, test_loader, std, epoch, config.batch_size)
            if early_stopping.early_stop:
                print('Early stopping...')
                break
            print(f'Progress: {epoch}/{config.num_epochs}/{loss:.5f}/{valid_error:.5f}/{test_error:.5f}')
            wandb.log({
                'epoch': epoch,
                'train_loss': loss,
                'valid_error': valid_error,
                'test_error': test_error
            })
        dist.barrier()

        if early_stopping is not None:
            early_stop = torch.tensor([early_stopping.early_stop], device=rank)
        else:
            early_stop = torch.tensor([False], device=rank)
        dist.broadcast(early_stop, src=0)
        if early_stop.item():
            break
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs!")
    time_start = time.time()
    args = (world_size, config)
    mp.spawn(run, args=args, nprocs=world_size, join=True)
    time_end = time.time()
    print(f'Total time: {time_end - time_start:.2f} seconds')
