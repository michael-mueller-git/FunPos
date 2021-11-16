
checkpoints_dir='checkpoint'
train_dir='./data/train'
skip_frames = 2
batch_size=1 # data loader is only implemented for case = 1!
img_width=128
img_height=128
img_channels=3
convlstm_hidden_dim=64
seq_len=8 # input seqence len
IMAGE_MEAN = [0.0, 0.0, 0.0]
