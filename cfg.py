gpu = 0
epsilon = 1e-8
gamma = 2
alpha = 0.25

# ISN and FEN Loss parameter
l_dice_alpha = 10.
l_wbec_alpha = 1.
l_lap_alpha = 50.

# FN Loss parameter
lfn_alpha = 100.

# rec loss
rec_lr_weight = 1.
lf_rec = 1.

# train
learning_rate = 1e-4 
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999 
max_iter = 5000000
max_epoch = 10000
show_loss_interval = 500
write_log_interval = 100
save_ckpt_interval = 25000
gen_example_interval = 1000
gen_example_tensor_interval = 200
with_BIN = False
with_real_data = True
with_recognizer = True
train_recognizer = True
experiment_name = 'finetune_large_0829'

#checkpoint_save_direction
checkpoint_savedir = './checkpoints'
loss_txt = './loss'

#pretrained_path
ckpt_path = './checkpoints/train_step-500000.model'
rec_ckpt_path = './rec_checkpoints/TPS-ResNet-BiLSTM-Attn-Seed1111_LP2024syn_dot/best_accuracy.pth'
clip_pretrained = './SPG_CLIP_textAdain/pretrained/clip/ViT-B-16.pt'

#Finetune_ckeckpoints
finetune_ckpt_path = './checkpoints/train_step-525000.model'
finetune_ckpt_savedir = './checkpoints/'
save_epoch = 10
predict_image = 5

# train_path
data_dir = './contrast_150k_LMDB'


# validation path
test_data_dir = './Real_data/LP2024/train'
temp_data_dir = './checkpoints/finetune_large_0829/test_data'


finetune_dir = './'


# dataloader
batch_size = 16
real_bs = 2
with_real_data = True if real_bs > 0 else False
data_shape = [128, 128] 
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
mask_s_dir = 'mask_s'
mask_ts_dir = 'mask_ts'

chardict = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789·'
# chardict = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789·-' # modify for 5 num lp

MAX_CHARS = 12
MAX_LEN = MAX_CHARS + 2
