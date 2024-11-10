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
from loss import build_ISN_loss, build_discriminator_loss, build_FN_loss_rec
from data_loader import Dataloader_SPG,test_dataset_synth,Dataloader_SPG_combine,test_dataset_synth_combine
from torch.utils.data import DataLoader
from SPG_Module.SSN.TRGAN import TRGAN
from SPG_Module.FN.StarGAN import StarGAN
from SPG_Module.Discriminator.Discriminator import Discriminator
from SPG_Module.Perceptual_Model.VGG19 import Vgg19
from SPG_Module.Rec.rec_model import Rec_Model
from SPG_Module.Rec.rec_utils import AttnLabelConverter
from transformers import SegformerModel, SegformerConfig
from tensorboardX import SummaryWriter
import cv2
from PIL import Image


configuration = SegformerConfig()
configuration.num_labels = 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-1, 1) 

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    train_time = get_train_name()

    experiment_name = cfg.experiment_name

    writer = SummaryWriter('runs/' + train_time[:8] + '_' + experiment_name)
    # writer = SummaryWriter('/media/avlab/disk2/SPG_LP2024_Jimmy_cross/runs/20231208_low_contrast')

    loss_txt = os.path.join(cfg.loss_txt, experiment_name + '_' + train_time[4:8] + '.txt')

    with open(loss_txt, 'w')as fw:
      fw.writelines('experiment_name:' + '\t' + experiment_name + '\n' + 
                    'checkpoint_savedir:' + '\t' + cfg.checkpoint_savedir + '\n' + 
                    'test data:' + '\t' + cfg.temp_data_dir)
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    
    train_data = Dataloader_SPG_combine(cfg)
    
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = True, collate_fn = custom_collate,  pin_memory = True)
    
    example_data2 = test_dataset_synth_combine(cfg,transform = None)
        
    example_loader2 = DataLoader(dataset = example_data2, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate_test)
    
    print_log('training start.', content_color = PrintColor['yellow'])
        
    ISN =  TRGAN(in_channels = 3).cuda()
    FN  =  StarGAN(in_channels = 3).cuda()
    D_fn = Discriminator(in_channels = 6).cuda()
        
    vgg_features = Vgg19().cuda()

    #Recogntion
    if cfg.with_recognizer:
        converter = AttnLabelConverter('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789·')
        Recognizer = Rec_Model(cfg)
        rec_state_dict = torch.load(cfg.rec_ckpt_path, map_location='cpu')
        if len(rec_state_dict) == 1:
            rec_state_dict = rec_state_dict['recognizer']
        rec_state_dict = {k.replace('module.', ''): v for k, v in rec_state_dict.items()}
        Recognizer.cuda()
        Recognizer.load_state_dict(rec_state_dict)
        print_log('Recognizer module loaded: {}'.format(cfg.rec_ckpt_path)) 
   
    ISN_solver = torch.optim.Adam(ISN.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    FN_solver = torch.optim.Adam(FN.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D_fn_solver = torch.optim.Adam(D_fn.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    if cfg.with_recognizer and cfg.train_recognizer:
        Rec_solver = torch.optim.Adam(Recognizer.parameters(), lr=cfg.rec_lr_weight * cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


    print(cfg.data_dir)
    try:
      checkpoint  = torch.load(cfg.ckpt_path)

      ISN.load_state_dict(checkpoint['Input style network'], strict=False)  
      FN.load_state_dict(checkpoint['generator_FN'], strict=False)                 
      
      if 'steps' in checkpoint:
        resume_step = checkpoint['steps']

      print('Resuming after loading...')

    except FileNotFoundError:

      print('checkpoint not found')
      resume_step = 0
      pass          
    
    trainiter = iter(train_data)
    example_iter2 = iter(example_loader2)
    # resume_step = 0
    for step in tqdm(range(resume_step,cfg.max_iter),total = cfg.max_iter,initial=resume_step):
     
        if ((step+1) % cfg.save_ckpt_interval == 0) and (step+1)!=resume_step+1:
            if not os.path.exists(cfg.checkpoint_savedir):
              os.makedirs(cfg.checkpoint_savedir) 
            torch.save(

                {
                    'Input style network': ISN.state_dict(), 
                    'generator_FN'       : FN.state_dict(),
                    'discriminator_fn'   : D_fn.state_dict(),
                    'ISN_optimizer'      : ISN_solver.state_dict(),
                    'FN_optimizer'       : FN_solver.state_dict(),
                    'd_fn_optimizer'     : D_fn_solver.state_dict(),
                    'steps': step,
                },
                os.path.join(cfg.checkpoint_savedir,f'train_step-{step+1}.model'),
            )
            if cfg.with_recognizer:
                torch.save({'recognizer': Recognizer.state_dict()}, cfg.checkpoint_savedir + 'best_recognizer.model')

        
        try:
          
          i_ts, i_tt, i_s, t_b, t_f, mask_t, mask_s, texts = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_ts, i_tt, i_s, t_b, t_f, mask_t, mask_s, texts = trainiter.next()

        D_fn_solver.zero_grad()

        i_ts = i_ts.cuda()
        i_tt = i_tt.cuda()
        i_s = i_s.cuda()
        t_b = t_b.cuda()
        t_f = t_f.cuda()
        mask_t = mask_t.cuda()
        mask_s = mask_s.cuda()

        if cfg.with_recognizer:
            texts, texts_length = converter.encode(texts, batch_max_length=34)
            texts = texts.cuda()
            rec_target = texts[:, 1:]
            labels = [t_b, t_f, mask_t, mask_s, rec_target]
        else:
            labels = [t_b, t_f, mask_t, mask_s]
         
        requires_grad(ISN, False)
        requires_grad(FN, False)
        requires_grad(D_fn,True)# hat(I)_t discriminator
        # breakpoint()
        s_sk, t_sk= ISN(i_ts, i_tt, i_s)


        input_mask = 1 - s_sk
        input = i_s * input_mask
        input_mask = torch.cat((input_mask, input_mask, input_mask), 1)

        output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
        o_b = output

        o_f = FN(o_b,t_sk,i_s)

        i_df_true = torch.cat((t_f,i_s), dim=1)
        i_df_pred = torch.cat((o_f,i_s), dim=1)

        o_df_true = D_fn(i_df_true)
        o_df_pred = D_fn(i_df_pred)

        df_loss = build_discriminator_loss(o_df_true, o_df_pred)
        df_loss.backward()
        D_fn_solver.step()
        
        if ((step+1) % 2 == 0):     

            requires_grad(ISN, True)
            requires_grad(FN, True)
            requires_grad(D_fn,False)

            ISN_solver.zero_grad()

            if cfg.with_recognizer and cfg.train_recognizer:
              Rec_solver.zero_grad()
            
            s_sk, t_sk= ISN(i_ts, i_tt, i_s) 

            s_loss, s_detail = build_ISN_loss(s_sk, mask_s)
            t_loss, t_detail = build_ISN_loss(t_sk, mask_t)

            total_loss = s_loss + t_loss
            total_loss.backward() 
            ISN_solver.step()
            
            s_sk = s_sk.detach()
            t_sk = t_sk.detach()
            
            input_mask = 1 - s_sk
            input = i_s * input_mask
            input_mask = torch.cat((input_mask, input_mask, input_mask), 1)
            output = input_mask * input + (1 - input_mask) * torch.normal(float(torch.mean(input)),float(torch.std(input)),size=input.size()).cuda()
            o_b = output

            o_f = FN(o_b,t_sk,i_s)

            i_df_pred = torch.cat((o_f,i_s), dim=1)

            o_df_pred = D_fn(i_df_pred)

            fn_vgg = torch.cat((t_f, o_f), dim = 0)

            out_fn_vgg = vgg_features(fn_vgg)

            if cfg.with_recognizer:
                # if cfg.use_rgb:
                #     tmp_o_f = o_f
                #     tmp_t_f = t_f
                # else:
                #     tmp_o_f = rgb2grey(o_f)
                #     tmp_t_f = rgb2grey(t_f)
                rec_preds = Recognizer(o_f, texts[:, :-1], is_train=False)
                out_g = [o_f, rec_preds]
            else:
                out_g = [o_f]

            # fn_loss, fn_detail = build_FN_loss(o_f, t_f, out_fn_vgg, o_df_pred)
            fn_loss, fn_detail = build_FN_loss_rec(out_g, labels, out_fn_vgg, o_df_pred)
            # print(fn_detail[4])

            FN_solver.zero_grad()     
            fn_loss.backward()
            FN_solver.step()

            if cfg.with_recognizer and cfg.train_recognizer:
              Rec_solver.step()

            total_loss_list = [s_loss,t_loss,fn_loss]

            # SPG_list_nams = ['ISN_s','ISN_t','FN']
            # ISN_s_list_name = ['l_t_sk','l_t_sk_BCE','l_t_lap']
            # ISN_t_list_name = ['l_t_sk','l_t_sk_BCE','l_t_lap']

            # for i, (spg, isn_s, isn_t) in enumerate(zip(SPG_list_nams, ISN_s_list_name, ISN_t_list_name)):
            #   print('SPG/' + spg, total_loss_list[i].item(), step + 1)
            #   print('ISN_s/' + isn_s, s_detail[i].item(), step + 1)
            #   print('ISN_t/' + isn_t, t_detail[i].item(), step + 1)

            
        if ((step+1) % cfg.show_loss_interval == 0):

          ### print loss
          if cfg.with_BIN == True:
            print('Iter: {}/{} | ISN_s: {} | ISN_t: {} | FN: {}'.format(step+1, cfg.max_iter, s_loss.item(), t_loss.item(),fn_loss.item()))
          else:
            print('Iter: {}/{} | ISN_s: {} | ISN_t: {} | FN: {}'.format(step+1, cfg.max_iter, s_loss.item(), t_loss.item(),fn_loss.item()))
                        
        ### Write to tensorboard
        if ((step+1) % cfg.write_log_interval == 0):
            
            SPG_list_nams = ['ISN_s','ISN_t','FN']
            ISN_s_list_name = ['l_t_sk','l_t_sk_BCE','l_t_lap']
            ISN_t_list_name = ['l_t_sk','l_t_sk_BCE','l_t_lap']

            if cfg.with_BIN == True:
              BIN_list_name = ['l_b_l2', 'l_b_gan', 'l_bg_vgg_per', 'l_bg_vgg_style']

            # FN_list_name =['l_fn_l2', 'l_fn_vgg_per', 'l_fn_vgg_style', 'o_df_pred']
            if cfg.with_recognizer:
                FN_list_name =['l_fn_l2', 'l_fn_vgg_per', 'l_fn_vgg_style', 'l_fn_gan', 'l_f_rec']
            else:
                FN_list_name =['l_fn_l2', 'l_fn_vgg_per', 'l_fn_vgg_style', 'l_fn_gan']
            
            for i in range(len(total_loss_list)):
              writer.add_scalar('SPG/'+SPG_list_nams[i], total_loss_list[i].item(), step+1)

            for i in range(len(s_detail)):
              writer.add_scalar('ISN_s/'+ISN_s_list_name[i], s_detail[i].item(), step+1)

            for i in range(len(t_detail)):
              writer.add_scalar('ISN_t/'+ISN_t_list_name[i], t_detail[i].item(), step+1)

            for i in range(len(fn_detail)):
              
              writer.add_scalar('FN/'+FN_list_name[i], fn_detail[i].item(), step+1)

            with open(loss_txt, 'a+')as fw:
              fw.writelines('Iter: {}/{} | ISN_s: {} | ISN_t: {} | FN: {}'.format(step+1, cfg.max_iter, s_loss.item(), t_loss.item(),fn_loss.item())+ '\n')
              # fw.writelines('SPG/'+SPG_list_nams[i], total_loss_list[i].item(), step+1)
              
        ### predict ###
        if ((step+1) % cfg.gen_example_tensor_interval == 0):
           
            with torch.no_grad():

                try:

                  i_ts, i_tt, i_s, img_name_batch, padding_list_batch,img_shape_batch = example_iter2.next()
                
                except StopIteration:

                  example_iter2 = iter(example_loader2)

                  i_ts, i_tt, i_s, img_name_batch, padding_list_batch,img_shape_batch = example_iter2.next()                
                
                i_ts = i_ts.cuda()
                i_tt = i_tt.cuda()
                i_s = i_s.cuda()
            
                # o_m,angle_list = FEN(i_s)
                # o_m = o_m[0] 
                # max_pool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
                # o_m = max_pool(o_m)

                s_sk, t_sk= ISN(i_ts, i_tt, i_s)   

                # if cfg.with_BIN == True:
                #   o_b,o_bg_fu = BIN(i_s,o_m)
                # else:
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

                for i in range(len(img_name_batch)):
                  name = str(img_name_batch[i])
                  padding_list = padding_list_batch[i]
                  img_shape = img_shape_batch[i]

                  # o_sk = o_sk.squeeze(0).detach().to('cpu')
                  s_sk  =  s_sk_batch[i].transpose(1,2,0)
                  t_sk = t_sk_batch[i].transpose(1,2,0)
                  o_b  = o_b_batch[i].transpose(1,2,0)
                  o_f  = o_f_batch[i].transpose(1,2,0)                  

                  o_b  = (o_b*255).astype(int)
                  o_f  = (o_f*255).astype(int)
                  t_sk = (t_sk*255).astype(int)
                  s_sk  = (s_sk*255).astype(int)

                  o_b  = resize_back(o_b,padding_list,img_shape).astype(int)[...,[2,1,0]]
                  o_f  = resize_back(o_f,padding_list,img_shape).astype(int)[...,[2,1,0]]
                  t_sk = resize_back(t_sk,padding_list,img_shape)
                  s_sk  = resize_back(s_sk,padding_list,img_shape)

                  if not os.path.exists(cfg.temp_data_dir):
                    os.makedirs(cfg.temp_data_dir)
                  # print('hi')

                  cv2.imwrite(os.path.join(cfg.temp_data_dir,name.replace('.','_tsk.')),t_sk)
                  cv2.imwrite(os.path.join(cfg.temp_data_dir,name.replace('.','_ssk.')),s_sk)
                  cv2.imwrite(os.path.join(cfg.temp_data_dir,name.replace('.','_bg.')),o_b)
                  cv2.imwrite(os.path.join(cfg.temp_data_dir,name.replace('.','_fn.')),o_f)
                  
                  t_sk2 = cv2.imread(os.path.join(cfg.temp_data_dir,name.replace('.','_tsk.')), cv2.IMREAD_GRAYSCALE)
                  s_sk2  = cv2.imread(os.path.join(cfg.temp_data_dir,name.replace('.','_ssk.')), cv2.IMREAD_GRAYSCALE)
                  o_b2  = cv2.imread(os.path.join(cfg.temp_data_dir,name.replace('.','_bg.')))
                  o_f2  = cv2.imread(os.path.join(cfg.temp_data_dir,name.replace('.','_fn.')))
                  # o_t_c2 = cv2.imread(os.path.join(cfg.temp_data_dir,name.replace('.','_t_c.')))
                  # print(os.path.join(cfg.temp_data_dir,name.replace('.','_fn.')))

                  writer.add_image('sample'+name+'/'+name.replace('.','_tsk.'),t_sk2,step+1,dataformats='HW')
                  writer.add_image('sample'+name+'/'+name.replace('.','_ssk.'),s_sk2,step+1,dataformats='HW')
                  writer.add_image('sample'+name+'/'+name.replace('.','_bg.'),o_b2[...,[2,1,0]],step+1,dataformats='HWC')
                  writer.add_image('sample'+name+'/'+name.replace('.','_fn.'),o_f2[...,[2,1,0]],step+1,dataformats='HWC')
                  # writer.add_image('sample'+name+'/'+name.replace('.','_t_c.'),o_t_c2[...,[2,1,0]],step+1,dataformats='HWC')

                
if __name__ == '__main__':
    main()
