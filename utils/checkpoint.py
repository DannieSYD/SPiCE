import os
import uuid
import torch


def generate_checkpoint_filename(extension='pt'):
    unique_id = uuid.uuid4()
    filename = f'checkpoint_{unique_id}.{extension}'
    return filename


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    conf_encoder_state_dict = {k.replace('graph_encoders.', ''): v for k, v in model_state_dict.items()
                               if 'graph_encoders' in k}

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # model.module.load_state_dict(model_state_dict)
        model.module.graph_encoders.load_state_dict(conf_encoder_state_dict)
    else:
        # model.load_state_dict(model_state_dict)
        model.graph_encoders.load_state_dict(conf_encoder_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Checkpoint at epoch {epoch} loaded!")

    return model, optimizer, scheduler, epoch


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
