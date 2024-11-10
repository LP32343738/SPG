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
from SPG_Module.Discriminator.Discriminator import Discriminator
from SPG_Module.Perceptual_Model.VGG19 import Vgg19
#from model import Input_conversion_net
from torchvision import models, transforms, datasets
from loss import build_discriminator_loss,build_SPG_loss
from data_loader import test_dataset_finetune_combine_self,dataloader_fintune_real_combine
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerModel, SegformerConfig
from tensorboardX import SummaryWriter
import cv2
from PIL import Image
import shutil



configuration = SegformerConfig()
configuration.num_labels = 1




def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-1, 1) 

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    train_time = get_train_name()

    experiment_name = cfg.experiment_name

    writer = SummaryWriter('runs/' + train_time[:8] + '_' + experiment_name)

    loss_txt = os.path.join(cfg.loss_txt, experiment_name + '_' + train_time[4:8] + '.txt')
    
    with open(loss_txt, 'a+')as fw:
      fw.writelines('\n' +'Time:' + '\t' + train_time + '\n' + '\n' + 
                    'Experiment_name:' + '\t' + experiment_name + '\n' + '\n' + 
                    'Checkpoint_path:' + '\t' + cfg.finetune_ckpt_path + '\n' + '\n' + 
                    'Finetune_ckpt_savedir:' + '\t' + cfg.finetune_ckpt_savedir + '\n' + '\n' +
                    'Test data:' + '\t' + cfg.temp_data_dir + '\n' )
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    
    print_log('training start.', content_color = PrintColor['yellow'])  



    ISN =  TRGAN(in_channels = 3).cuda()
    # FEN =  SegformerForSemanticSegmentation(configuration).cuda()
    FN  =  StarGAN(in_channels = 3).cuda()
    D_SPG = Discriminator(in_channels = 6).cuda()

        
    vgg_features = Vgg19().cuda()    
    SPG_solver = torch.optim.Adam(list(ISN.parameters()) + list(FN.parameters()), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))    
    D_SPG_solver = torch.optim.Adam(D_SPG.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    

    print(cfg.finetune_dir)
    try:
      checkpoint  = torch.load(cfg.finetune_ckpt_path)

      ISN.load_state_dict(checkpoint['Input style network'], strict=False)       
      FN.load_state_dict(checkpoint['generator_FN'], strict=False)
      
      if 'steps' in checkpoint:
        resume_step = checkpoint['steps']

      print('Resuming after loading...')

    except FileNotFoundError:

      print('checkpoint not found')
      resume_step = 0
      pass          
    
    # trainiter2 = iter(train_data2)
    # example_iter2 = iter(example_loader2)

    step = 0
    for epoch in range(cfg.max_epoch):

      if ((epoch+1) % cfg.save_epoch == 0):
            if not os.path.exists(cfg.finetune_ckpt_savedir):
              os.makedirs(cfg.finetune_ckpt_savedir) 
            torch.save(
                {
                    'Input style network': ISN.state_dict(),
                    'generator_FN': FN.state_dict(),
                    'discriminator_SPG':D_SPG.state_dict(),
                    'steps': step,
                },
                cfg.finetune_ckpt_savedir+f'train_step-{step+1}.model',
            )
      ### Predict image ###
      if (epoch + 1) % cfg.predict_image == 0 :
        with torch.no_grad():

          if not os.path.exists(os.path.join(cfg.finetune_dir, cfg.t_f_dir)):
            os.makedirs(os.path.join(cfg.finetune_dir, cfg.t_f_dir))
          
          example_data2 = test_dataset_finetune_combine_self(cfg,transform = None)
          example_loader2 = DataLoader(dataset = example_data2, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate_test_combine)    
          valloop = tqdm(enumerate(example_loader2),total=len(example_loader2))
          a = 0
          for iter,valiter2 in valloop:

            i_ts, i_tt, i_s, img_name_batch, padding_list_batch,img_shape_batch = valiter2
                              
            i_ts = i_ts.cuda()
            i_tt = i_tt.cuda()
            i_s = i_s.cuda()
            
            s_sk, t_sk= ISN(i_ts, i_tt, i_s)  
            max_pool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
            s_sk1 = max_pool(s_sk)
            s_sk1 = max_pool(s_sk1)
            t_sk1 =  max_pool(t_sk)
            t_sk1 =  max_pool(t_sk1)
            o_m = (s_sk1 + t_sk1)/2
            o_m = (o_m>0.4).float()


            input_mask = 1 - o_m
            input = i_s * input_mask
            input_mask = torch.cat((input_mask, input_mask, input_mask), 1)
            output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
            o_b = output

            o_f = FN(o_b,t_sk,i_s)
            
            o_m_batch =  o_m.detach().to('cpu').numpy()
            t_sk_batch = t_sk.detach().to('cpu').numpy()
            o_b_batch = o_b.detach().to('cpu').numpy()
            o_f_batch = o_f.detach().to('cpu').numpy()

            for i in range(len(img_name_batch)):
              # gt = i_t_batch[i]
              name = str(img_name_batch[i])                     
              padding_list = padding_list_batch[i]
              img_shape = img_shape_batch[i]

              o_m =  o_m_batch[i].transpose(1,2,0)
              t_sk = t_sk_batch[i].transpose(1,2,0)
              o_b = o_b_batch[i].transpose(1,2,0)
              o_f = o_f_batch[i].transpose(1,2,0)
              

              o_b = (o_b*255).astype(int)

              o_f = (o_f*255).astype(int)

              t_sk = (t_sk*255).astype(int)

              o_m = (o_m*255).astype(int)

              o_b = resize_back(o_b,padding_list,img_shape).astype(int)[...,[2,1,0]]
              o_f = resize_back(o_f,padding_list,img_shape).astype(int)[...,[2,1,0]]
              t_sk = resize_back(t_sk,padding_list,img_shape)
              o_m = resize_back(o_m,padding_list,img_shape)

              temp_data_dir = cfg.temp_data_dir + '_finetune'

              if not os.path.exists(temp_data_dir):
                os.makedirs(temp_data_dir)
           
              if name.split('.')[-1] != 'png':
                name = name.replace(name.split('.')[-1],'png')
              
              cv2.imwrite(os.path.join(temp_data_dir,name.replace('.png','_tsk.png')),t_sk)
              cv2.imwrite(os.path.join(temp_data_dir,name.replace('.png','_ssk.png')),o_m)
              cv2.imwrite(os.path.join(temp_data_dir,name.replace('.png','_bg.png')),o_b)
              cv2.imwrite(os.path.join(temp_data_dir,name.replace('.png','_fn.png')),o_f)
              cv2.imwrite(os.path.join(cfg.finetune_dir, cfg.t_f_dir,name),o_f)
                     

              t_sk2 = cv2.imread(os.path.join(temp_data_dir,name.replace('.png','_tsk.png')), cv2.IMREAD_GRAYSCALE)
              s_sk2 = cv2.imread(os.path.join(temp_data_dir,name.replace('.png','_ssk.png')), cv2.IMREAD_GRAYSCALE)
              o_b2 = cv2.imread(os.path.join(temp_data_dir,name.replace('.png','_bg.png')))
              o_f2 = cv2.imread(os.path.join(temp_data_dir,name.replace('.png','_fn.png')))
            
              writer.add_image('sample'+name+'/'+name.replace('.png','_tsk.png'),t_sk2,step+1,dataformats='HW')
              writer.add_image('sample'+name+'/'+name.replace('.png','_ssk.png'),s_sk2,step+1,dataformats='HW')
              writer.add_image('sample'+name+'/'+name.replace('.png','_bg.png'),o_b2[...,[2,1,0]],step+1,dataformats='HWC')
              writer.add_image('sample'+name+'/'+name.replace('.png','_fn.png'),o_f2[...,[2,1,0]],step+1,dataformats='HWC')

      train_data2 = dataloader_fintune_real_combine(cfg)
      train_data2 = DataLoader(dataset = train_data2,batch_size = cfg.batch_size, shuffle = True, collate_fn = custom_collate_finetune_cycle,  pin_memory = True)
      trainloop  = tqdm(enumerate(train_data2),total=len(train_data2))

      for iter,trainiter2 in trainloop:
     
          i_ts, i_s, t_f, i_tt = trainiter2           

          D_SPG_solver.zero_grad()
 
          i_ts = i_ts.cuda()
          i_s = i_s.cuda()
          t_f = t_f.cuda()
          i_tt = i_tt.cuda()
          
          # requires_grad(FEN, False)
          requires_grad(ISN, False)
          requires_grad(FN, False)
          requires_grad(D_SPG,True)

          # o_m,angle_list = FEN(i_s)
          # o_m = o_m[0]

          s_sk, t_sk= ISN(i_tt, i_ts, t_f)

          input_mask = 1 - s_sk
          input = t_f * input_mask
          input_mask = torch.cat((input_mask, input_mask, input_mask), 1)
          output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
          o_b = output

          o_f = FN(o_b,t_sk,t_f)

          i_df_true = torch.cat((t_f,i_s), dim=1)
          i_df_pred = torch.cat((o_f,i_s), dim=1)

          o_df_true = D_SPG(i_df_true)
          o_df_pred = D_SPG(i_df_pred)

          df_loss = build_discriminator_loss(o_df_true, o_df_pred)
          df_loss.backward()
          D_SPG_solver.step()
          
          # requires_grad(FEN, True)
          requires_grad(ISN, True)
          requires_grad(FN, True)
          requires_grad(D_SPG,False)                                

          # o_m,angle_list = FEN(i_s)
          # o_m = o_m[0]

          s_sk, t_sk= ISN(i_tt, i_ts, t_f)

          input_mask = 1 - s_sk
          input = t_f * input_mask
          input_mask = torch.cat((input_mask, input_mask, input_mask), 1)
          output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
          o_b = output

          o_f = FN(o_b, t_sk, t_f)

          i_df_pred = torch.cat((o_f,t_f), dim=1)

          o_df_pred = D_SPG(i_df_pred)
          
          fn_vgg = torch.cat((i_s, o_f), dim = 0)

          out_fn_vgg = vgg_features(fn_vgg)        

          SPG_loss, SPG_detail = build_SPG_loss(o_f, i_s, out_fn_vgg, o_df_pred)    

          SPG_solver.zero_grad()     
          SPG_loss.backward()
          SPG_solver.step()

          if ((step+1) % cfg.write_log_interval == 0):

            SPG_list_name =['l_fn_l1', 'l_fn_vgg_per', 'l_fn_vgg_style', 'o_df_pred']

            for i in range(len(SPG_detail)):
              writer.add_scalar('SPG/'+SPG_list_name[i], SPG_detail[i].item(), step+1)
          
          step = step +1
          trainloop.set_description(f'Epoch [{epoch}/{cfg.max_epoch}]')
          trainloop.set_postfix(G_loss = SPG_loss.item(),D_loss = df_loss.item())
      
      with open(loss_txt, 'a+')as fw:
         fw.writelines('Epoch: {}/{} | SPG: {}  '.format(epoch, cfg.max_epoch, SPG_loss.item())+ '\n')
      print('Epoch: {}/{} | SPG: {}  '.format(epoch, cfg.max_epoch, SPG_loss.item()))              
            
      
                    
if __name__ == '__main__':
    main()
