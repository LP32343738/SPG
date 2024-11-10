import os
from datetime import datetime
import numpy as np
import cv2
import cfg 
from skimage.transform import resize
from skimage import io
import torch

PrintColor = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'amaranth': 35,
    'ultramarine': 36,
    'white': 37
}

PrintStyle = {
    'default': 0,
    'highlight': 1,
    'underline': 4,
    'flicker': 5,
    'inverse': 7,
    'invisible': 8
}


def get_train_name():
    
    return datetime.now().strftime('%Y%m%d%H%M%S')


def print_log(s, time_style = PrintStyle['default'], time_color = PrintColor['blue'],
                content_style = PrintStyle['default'], content_color = PrintColor['white']):
    
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print (log)
    
def custom_collate_test(batch):
    
    i_ts_batch, i_tt_batch, i_s_batch, = [], [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []
    mask_s_batch = []
    img_name_batch = []
    padding_list_batch = []
    img_shape_batch = []
   
    for item in batch:
        
        i_ts, i_tt, i_s, img_name = item
        img_shape = i_s.shape[:2]
        padding_list,to_scale = check_image(i_s)

        i_s = resize(i_s, to_scale, preserve_range=True)        

        i_s = padding_image(i_s,padding_list)

        i_s = i_s.transpose((2, 0, 1))

        i_ts_batch.append(i_ts)
        i_tt_batch.append(i_tt) 
        i_s_batch.append(i_s)
        img_name_batch.append(img_name)
        padding_list_batch.append([padding_list,to_scale])
        img_shape_batch.append(img_shape)


    i_ts_batch = np.stack(i_ts_batch)
    i_tt_batch = np.stack(i_tt_batch)
    i_s_batch = np.stack(i_s_batch)
    

    i_ts_batch    = torch.from_numpy(i_ts_batch.astype(int))
    i_tt_batch    = torch.from_numpy(i_tt_batch.astype(int))
    i_s_batch    = torch.from_numpy(i_s_batch.astype(np.float32) / 255.) 

      
    return [i_ts_batch, i_tt_batch, i_s_batch, img_name_batch, padding_list_batch, img_shape_batch]

def custom_collate_test_combine(batch):
    
    i_ts_batch, i_tt_batch, i_s_batch, = [], [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []
    mask_s_batch = []
    img_name_batch = []
    padding_list_batch = []
    img_shape_batch = []
   
    for item in batch:
        
        i_ts, i_tt, i_s, img_name = item
        img_shape = i_s.shape[:2]
        padding_list,to_scale = check_image(i_s)

        i_s = resize(i_s, to_scale, preserve_range=True)        

        i_s = padding_image(i_s,padding_list)

        i_s = i_s.transpose((2, 0, 1))

        i_ts_batch.append(i_ts)
        i_tt_batch.append(i_tt) 
        i_s_batch.append(i_s)
        img_name_batch.append(img_name)
        padding_list_batch.append([padding_list,to_scale])
        img_shape_batch.append(img_shape)


    i_ts_batch = np.stack(i_ts_batch)
    i_tt_batch = np.stack(i_tt_batch)
    i_s_batch = np.stack(i_s_batch)
    

    i_ts_batch    = torch.from_numpy(i_ts_batch.astype(int))
    i_tt_batch    = torch.from_numpy(i_tt_batch.astype(int))
    i_s_batch    = torch.from_numpy(i_s_batch.astype(np.float32) / 255.) 

      
    return [i_ts_batch, i_tt_batch, i_s_batch, img_name_batch, padding_list_batch, img_shape_batch]

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    
def resize_back(image,padding_list,img_shape):
  if len(padding_list)==1:
     (padding_h,padding_w),to_scale = padding_list[0]
     (ori_h,ori_w)=img_shape[0]
     (h,w) = to_scale
  else:
    (padding_h,padding_w),to_scale = padding_list
    (ori_h,ori_w)=img_shape
    (h,w) = to_scale
  if image.shape[2] == 3:
    ori_img = np.zeros((h,w,3))
    ori_img[:,:,:] = image[padding_h:padding_h+h, padding_w:padding_w+w,:]
  else:
    ori_img = np.zeros((h,w,1))
    ori_img[:,:,:] = image[padding_h:padding_h+h, padding_w:padding_w+w,:]
  ori_img = cv2.resize(ori_img,(ori_w,ori_h)).astype(int)
  return ori_img


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def check_image(image):
    h, w = image.shape[:2]
    # pad_127 = np.ones((image.shape)) * 127
    if w>h :
      scale_ratio = cfg.data_shape[0] / w
      to_h = int(h* scale_ratio)
      to_h = int(round(to_h / 2)) * 2
      to_scale = (to_h, 128)
      padding_h = (128-to_h)//2
      padding_w = 0
      padding_list = (padding_h,padding_w)
    else:
      scale_ratio = cfg.data_shape[0] / h
      to_w = int(w * scale_ratio)
      to_w = int(round(to_w / 2)) * 2
      to_scale = (128, to_w)
      padding_w = (128-to_w)//2
      padding_h = 0
      padding_list = (padding_h,padding_w)
    
    return padding_list,to_scale
    


def padding_image(image,padding_list,pad_0=False):
  if pad_0 == False:
    pad_127 = np.ones((cfg.data_shape[0],cfg.data_shape[1],image.shape[2])) * 127
  else :
    pad_127 = np.zeros((cfg.data_shape[0],cfg.data_shape[1],image.shape[2]))
  padding_h,padding_w = padding_list
  h,w =image.shape[:2]
  pad_127[padding_h:padding_h+h, padding_w:padding_w+w,:] = image
  # cv2.imwrite('/home/avlab/scenetext/SRNet-master_endtoend_CCPD/test_image/test.jpg',pad_127)

  return pad_127

def custom_collate(batch):
    
    i_ts_batch, i_tt_batch, i_s_batch, t_b_batch, t_f_batch, mask_t_batch, mask_s_batch, texts_batch = [], [], [], [], [], [], [], []

    
    for item in batch:
        
        i_ts, i_tt, i_s, t_b, t_f, mask_t, mask_s, texts= item

        padding_list,to_scale = check_image(i_s)

        i_s = resize(i_s, to_scale, preserve_range=True)
        t_b = resize(t_b, to_scale, preserve_range=True)
        t_f = resize(t_f, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)
        mask_s = np.expand_dims(resize(mask_s, to_scale, preserve_range=True), axis = -1)
        

        i_s = padding_image(i_s,padding_list)
        t_b = padding_image(t_b,padding_list)
        t_f = padding_image(t_f,padding_list)
        mask_t = padding_image(mask_t,padding_list,pad_0=True)
        mask_s = padding_image(mask_s,padding_list,pad_0=True)



        i_s = i_s.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1)) 
        mask_s = mask_s.transpose((2, 0, 1)) 



        i_ts_batch.append(i_ts)
        i_tt_batch.append(i_tt)
        i_s_batch.append(i_s)
        t_b_batch.append(t_b)
        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)
        mask_s_batch.append(mask_s)
        texts_batch.append(texts)


    i_ts_batch = np.stack(i_ts_batch)
    i_tt_batch = np.stack(i_tt_batch)
    i_s_batch = np.stack(i_s_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.stack(mask_t_batch)
    mask_s_batch = np.stack(mask_s_batch)
    texts_batch = np.stack(texts_batch)




    i_ts_batch    = torch.from_numpy(i_ts_batch.astype(int))
    i_tt_batch    = torch.from_numpy(i_tt_batch.astype(int))
    i_s_batch    = torch.from_numpy(i_s_batch.astype(np.float32) / 255.) 
    t_b_batch    = torch.from_numpy(t_b_batch.astype(np.float32) / 255.)
    t_f_batch    = torch.from_numpy(t_f_batch.astype(np.float32) / 255.)

    mask_t_batch = torch.from_numpy(mask_t_batch.astype(np.float32)/ 255.) 
    mask_s_batch = torch.from_numpy(mask_s_batch.astype(np.float32) / 255.)
    texts_batch = texts_batch.tolist()

  

    return [i_ts_batch, i_tt_batch, i_s_batch, t_b_batch, t_f_batch, mask_t_batch , mask_s_batch, texts_batch]

def custom_collate_finetune(batch):
    
    i_t_batch, i_s_batch, t_f_batch = [], [],[]

    
    for item in batch:
        
        i_t, i_s, t_f= item

        padding_list,to_scale = check_image(i_s)

        i_s = resize(i_s, to_scale, preserve_range=True)
        t_f = resize(t_f, to_scale, preserve_range=True)
        

        i_s = padding_image(i_s,padding_list)
        t_f = padding_image(t_f,padding_list)


        i_s = i_s.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))


        i_t_batch.append(i_t) 
        i_s_batch.append(i_s)
        t_f_batch.append(t_f)


    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_f_batch = np.stack(t_f_batch)



    i_t_batch    = torch.from_numpy(i_t_batch.astype(int))
    i_s_batch    = torch.from_numpy(i_s_batch.astype(np.float32) / 255.) 
    t_f_batch    = torch.from_numpy(t_f_batch.astype(np.float32) / 255.)


    return [i_t_batch, i_s_batch, t_f_batch]

def custom_collate_finetune_cycle(batch):
    
    i_t_batch, i_s_batch, t_f_batch, t_t_batch = [], [],[], []

    
    for item in batch:
        
        i_t, i_s, t_f, t_t= item

        padding_list,to_scale = check_image(i_s)

        i_s = resize(i_s, to_scale, preserve_range=True)
        t_f = resize(t_f, to_scale, preserve_range=True)
        

        i_s = padding_image(i_s,padding_list)
        t_f = padding_image(t_f,padding_list)


        i_s = i_s.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))


        i_t_batch.append(i_t) 
        t_t_batch.append(t_t) 
        i_s_batch.append(i_s)
        t_f_batch.append(t_f)


    i_t_batch = np.stack(i_t_batch)
    t_t_batch = np.stack(t_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_f_batch = np.stack(t_f_batch)




    i_t_batch    = torch.from_numpy(i_t_batch.astype(int))
    t_t_batch    = torch.from_numpy(t_t_batch.astype(int))
    i_s_batch    = torch.from_numpy(i_s_batch.astype(np.float32) / 255.) 
    t_f_batch    = torch.from_numpy(t_f_batch.astype(np.float32) / 255.)


    return [i_t_batch, i_s_batch, t_f_batch, t_t_batch]
