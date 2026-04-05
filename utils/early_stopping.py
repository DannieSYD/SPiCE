from utils.checkpoint import save_checkpoint
import math


class EarlyStopping:
    def __init__(self, patience=100, verbose=False, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.nan_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        score = -val_loss

        if math.isnan(score):
            self.nan_counter += 1
            if self.nan_counter > 1:
                self.early_stop = True

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({-self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        save_checkpoint(self.path, model, optimizer, scheduler, epoch)
