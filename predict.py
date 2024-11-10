# Training script for the SRNet. Refer README for instructions.
# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from SPG_Module.FEN.modeling_segformer import SegformerForSemanticSegmentation
from SPG_Module.SSN.TRGAN import TRGAN
from SPG_Module.FN.StarGAN import StarGAN
from SPG_Module.BIN.MADF import MADF
from data_loader import demo_dataset, demo_dataset_combine
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerModel, SegformerConfig
from tensorboardX import SummaryWriter
import cv2
from PIL import Image


configuration = SegformerConfig()
configuration.num_labels = 1

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-1, 1) 

def makedirs(path):
  if not os.path.exists(path):
      os.makedirs(path)

import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = torch.nn.ZeroPad2d((0, 1, 1, 0))
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = '/media/avlab/91973954-b144-4aaf-8efc-932b487ff082/scenetext/LP_github/demo/i_s')
    parser.add_argument('--save_dir', help = 'Directory to save result', 
                          default = '/media/avlab/91973954-b144-4aaf-8efc-932b487ff082/scenetext/LP_github/demo/temp')
    parser.add_argument('--input_text', help = '',
               default='/media/avlab/91973954-b144-4aaf-8efc-932b487ff082/scenetext/LP_github/demo')
    parser.add_argument('--checkpoint', help = 'ckpt', 
                        default = '/media/avlab/91973954-b144-4aaf-8efc-932b487ff082/scenetext/LP_github/LP/train_step-28939.model')
    args = parser.parse_args()

    assert args.input_dir is not None
    assert args.save_dir is not None
    assert args.checkpoint is not None
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    
    print_log('model compiling start.', content_color = PrintColor['yellow'])
    
    example_data2 = demo_dataset_combine(args.input_dir,args.input_text,transform = None)
        
    example_loader2 = DataLoader(dataset = example_data2, batch_size = 16, shuffle = False, collate_fn = custom_collate_test_combine)
    example_iter2 = iter(example_loader2)
    # ISN = Input_style_Net(in_channels = 3).cuda()    
    ISN = TRGAN(in_channels = 3).cuda()
    # ISN_C = Input_style_Net_C(in_channels = 3).cuda()
    # FEN = SegformerForSemanticSegmentation(configuration).cuda()
    # FEN = Foreground_Extraction_Net(in_channels = 3).cuda()
    # BIN = MADF(in_channels = 3).cuda()
    FN =StarGAN(in_channels = 3).cuda()

    checkpoint = torch.load(args.checkpoint)
    # checkpoint2 = torch.load('/media/avlab/disk2/LP2022_checkpoint0330/train_step-647500.model')

    # FEN.load_state_dict(checkpoint['generator_FEN'], strict=False)
    ISN.load_state_dict(checkpoint['Input style network'], strict=False)
    # ISN_C.load_state_dict(checkpoint['generator_ISN_C'], strict=False)
    # BIN.load_state_dict(checkpoint['generator_BIN'], strict=False) 
    FN.load_state_dict(checkpoint['generator_FN'], strict=False) 

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])                
          
    with torch.no_grad():

      for step in tqdm(range(len(example_data2))):

        try:

          i_ts, i_tt, i_s, img_name_batch, padding_list_batch, img_shape_batch = example_iter2.next()
        
        except StopIteration:

          example_iter2 = iter(example_loader2)

          i_ts, i_tt, i_s, img_name_batch, padding_list_batch, img_shape_batch = example_iter2.next()                
        
        i_ts = i_ts.cuda()
        i_tt = i_tt.cuda()
        i_s = i_s.cuda()
    
  
        s_sk, t_sk= ISN(i_ts, i_tt, i_s)

        # o_t= ISN_C(i_s)

        # o_t_mask = o_t*o_sk
        
    
        # o_b,o_bg_fu = BIN(i_s,o_m)
        input_mask = 1 - s_sk
        input = i_s * input_mask
        input_mask = torch.cat((input_mask, input_mask, input_mask), 1)

        output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
        o_b = output

        o_f = FN(o_b,t_sk,i_s)

        
        s_sk_batch  =  s_sk.detach().to('cpu').numpy()
        t_sk_batch = t_sk.detach().to('cpu').numpy()
        o_b_batch  = o_b.detach().to('cpu').numpy()
        o_f_batch  = o_f.detach().to('cpu').numpy()

        for i in range(s_sk_batch.shape[0]):
          name = str(img_name_batch[i])
          padding_list = padding_list_batch[i]
          img_shape = img_shape_batch[i]

          # o_sk = o_sk.squeeze(0).detach().to('cpu')
          s_sk =  s_sk_batch[i].transpose(1,2,0)
          t_sk = t_sk_batch[i].transpose(1,2,0)
          # o_t = o_t_batch[i].transpose(1,2,0)
          o_b = o_b_batch[i].transpose(1,2,0)
          o_f = o_f_batch[i].transpose(1,2,0)

          # o_f = o_f*o_sk+o_b*(1-o_sk)
          # o_t_c = o_t_c_batch[i].transpose(1,2,0)
          
          # o_t = (((o_t+1)/2)*255).astype(int)

          # o_t_c = (((o_t_c+1)/2)*255).astype(int)

          o_b = o_b*255

          o_f = o_f*255

          t_sk = t_sk*255
          # o_sk = np.interp(o_sk,(o_sk.min(),o_sk.max()),(0,255))

          s_sk = s_sk*255
          # o_m = np.interp(o_m,(o_m.min(),o_m.max()),(0,255))

          # o_t = resize_back(o_t,padding_list,img_shape).astype(int)[...,[2,1,0]]
          o_b = resize_back(o_b,padding_list,img_shape).astype(int)[...,[2,1,0]]
          o_f = resize_back(o_f,padding_list,img_shape).astype(int)[...,[2,1,0]]
          t_sk = resize_back(t_sk,padding_list,img_shape)
          s_sk = resize_back(s_sk,padding_list,img_shape)
          # o_t_c = resize_back(o_t_c,padding_list,img_shape)

          # makedirs(os.path.join(args.save_dir,cfg.t_t_dir))
          makedirs(os.path.join(args.save_dir,cfg.t_sk_dir))
          makedirs(os.path.join(args.save_dir,cfg.mask_s_dir))
          makedirs(os.path.join(args.save_dir,cfg.t_b_dir))
          makedirs(os.path.join(args.save_dir,cfg.t_f_dir))
          # makedirs(os.path.join(args.save_dir,'t_t_c'))
          


          # cv2.imwrite(os.path.join(args.save_dir,cfg.t_t_dir,name),o_t)
          cv2.imwrite(os.path.join(args.save_dir,cfg.t_sk_dir,name),t_sk)
          cv2.imwrite(os.path.join(args.save_dir,cfg.mask_s_dir,name),s_sk)
          cv2.imwrite(os.path.join(args.save_dir,cfg.t_b_dir,name),o_b)
          cv2.imwrite(os.path.join(args.save_dir,cfg.t_f_dir,name),o_f)
          # cv2.imwrite(os.path.join(args.save_dir,'t_t_c',name),o_t_c)
          
      


                
if __name__ == '__main__':
    main()
