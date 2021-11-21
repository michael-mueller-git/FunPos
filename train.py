import torch
import os
import tqdm
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import Funscript_Dataset
from model.model1 import FunPosModel
from model.model2 import FunPosTransformerModel
from utils.config import CONFIG


MODEL = CONFIG['general']['select']
LR = CONFIG[MODEL]['lr']
LR_MILESTONES = CONFIG[MODEL]['lr_milestones']
EPOCHS = CONFIG[MODEL]['epochs']
MODEL_CLASS = CONFIG[MODEL]['class']
TRAIN_DIR = CONFIG['general']['train_dir']
BARCH_SIZE = CONFIG['general']['batch_size']
SEQ_LEN = CONFIG[MODEL]['seq_len']
CHEKPOINT_DIR = CONFIG['general']['checkpoint_dir']
MODEL_NAME = CONFIG[MODEL]['name']
SKIP_FRAMES = CONFIG[MODEL]['skip_frames']
IMG_WIDTH = CONFIG[MODEL]['img_width']
IMG_HEIGHT = CONFIG[MODEL]['img_height']


if __name__ == '__main__':
    train_dataset = Funscript_Dataset(
            data_dir = TRAIN_DIR,
            skip_frames = SKIP_FRAMES,
            seq_len = SEQ_LEN,
            img_width = IMG_WIDTH,
            img_height = IMG_HEIGHT,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BARCH_SIZE, shuffle=False, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exec('model = ' + MODEL_CLASS + '().to(device)')

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, LR_MILESTONES, gamma=0.1, last_epoch=-1)

    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        print('#'*80)
        print('Epoch {}/{}'.format(epoch, EPOCHS))

        train_size = len(train_dataset) // BARCH_SIZE
        train_epoch_loss = 0

        model.train()
        for param_group in scheduler.optimizer.param_groups:
            print('lr: %s'% param_group['lr'])

        pbar = tqdm.tqdm(train_dataloader, 'Train Epoch ' + str(epoch), ncols=80)
        for frames, pos in pbar:
            frames = frames.to(device)
            pos = pos.view(1, SEQ_LEN, 1).to(device) # convert to 3D tensor
            optimizer.zero_grad()
            out_pos = model(frames)
            loss = criterion(out_pos, pos)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        scheduler.step()
        train_epoch_loss /= train_size
        print('Train Loss', train_epoch_loss)

        if not os.path.exists(CHEKPOINT_DIR):
            os.mkdir(CHEKPOINT_DIR)

        check_path = os.path.join(CHEKPOINT_DIR, MODEL_NAME + '_ep_{:03d}.cp'.format(epoch))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, check_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
