import torch
import os
import tqdm
import config
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from dataset import Funscript_Dataset
from model.model import FunPosModel


LR = 1e-4
LR_MILESTONES = [7,12,15]
EPOCHS = 25


if __name__ == '__main__':
    train_dataset = Funscript_Dataset(config.train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FunPosModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, LR_MILESTONES, gamma=0.1, last_epoch=-1)

    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        print('#'*80)
        print('Epoch {}/{}'.format(epoch, EPOCHS))

        train_size = len(train_dataset) // config.batch_size
        train_epoch_loss = 0

        model.train()
        for param_group in scheduler.optimizer.param_groups:
            print('lr: %s'% param_group['lr'])

        pbar = tqdm.tqdm(train_dataloader, 'Train Epoch ' + str(epoch), ncols=80)
        for frames, pos in pbar:
            frames = frames.to(device)
            pos = pos.view(1, config.seq_len, 1).to(device) # convert to 3D tensor
            optimizer.zero_grad()
            out_pos = model(frames)
            loss = criterion(out_pos, pos)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        scheduler.step()
        train_epoch_loss /= train_size
        print('Train Loss', train_epoch_loss)

        if not os.path.exists(config.checkpoints_dir):
            os.mkdir(config.checkpoints_dir)

        check_path = os.path.join(config.checkpoints_dir, 'FunPos_ep_{:03d}'.format(epoch))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, check_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
