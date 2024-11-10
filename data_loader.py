# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
from skimage import io
from skimage.transform import resize,rotate
import numpy as np
import random
import cfg 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import lmdb
from PIL import Image, ImageFilter
import gen_plate_text
import shutil


class fore_dataloader_lmdb(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir

            env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.nSamples = int(txn.get('num-samples'.encode()))/8
            self.name_list = []
            for index in range(int(self.nSamples)):
                index += 1  # lmdb starts with 1
                imagename = 'image-%09d' % index
                self.name_list.append(imagename)

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
            with open (os.path.join(self.data_dir,'target_list.txt'),'r',encoding='utf-8') as f:
                self.text_list = f.readlines()
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:

            i_s = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.i_s_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            i_t = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.i_t_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)           
            t_t = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.t_t_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            t_sk = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.mask_t_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2GRAY)
            # t_f = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.t_f_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            # t_b = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.t_b_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)

        
        return [i_s, i_t, t_t, t_sk]
def angle_to_index(list):
    text_angle_list = []
    char_angle_list = []
    char_distance_list = []
    char_aspect_list = []
    for inf in list:
        text_angle = int(inf.split(' ')[1])
        char_angle = int(float(inf.split(' ')[2]))
        char_distance = int(float(inf.split(' ')[3]))
        char_aspect = float(inf.split(' ')[4].strip('\n'))
        text_angle_list.append(text_angle)
        char_angle_list.append(char_angle)
        char_distance_list.append(char_distance)
        char_aspect_list.append(char_aspect)
    text_angle_min,text_angle_max = min(text_angle_list),max(text_angle_list)
    char_angle_min,char_angle_max = min(char_angle_list),max(char_angle_list)
    char_distance_min,char_distance_max = min(char_distance_list),max(char_distance_list)
    char_aspect_min,char_aspect_max = min(char_aspect_list),max(char_aspect_list)
    text_angle_dict = {}
    A = 0
    for i in range(text_angle_min,text_angle_max+5,5):
        text_angle_dict.update({
            str(i) : A
        })
        A += 1
    B = 0
    char_angle_dict = {}
    for i in range(char_angle_min,char_angle_max+5,5):
        char_angle_dict.update({
            str(float(i)) : B
        })
        B += 1
    char_distance_dict = {} 
    C = 0
    for i in range(char_distance_min,char_distance_max+1,1):
        char_distance_dict.update({
            str(float(i)) : C
        })
        C += 1
    char_aspect_dict = {}
    D_0 = 0
    D = 0
    for i in range(0,116,1):
        if D_0%5 ==0 and D_0!=0:
            D += 1
        char_aspect_dict.update({
            str(i/10) : D
        })
        D_0 += 1
    return text_angle_dict,char_angle_dict,char_distance_dict,char_aspect_dict


# LABEL = ["\u00B7",'\u7696','\u6CAA','\u6D25','\u6E1D','\u5180','\u664B','\u8499','\u8FBD','\u5409',"\u9ED1","\u82CF","\u6D59",'\u4EAC','\u95FD','\u8D63','\u9C81', "\u8C6B","\u9102","\u6E58","\u6FB3","\u6842","\u743C",'\u5DDD','\u8D35','\u4E91','\u85CF','\u9655','\u7518','\u9752','\u5B81','\u65B0','\u8B66','\u5B66','\u0041','\u0042','\u0043','\u0044','\u0045','\u0046','\u0047','\u0048','\u004A','\u004B','\u004C','\u004D','\u004E','\u0050','\u0051','\u0052','\u0053','\u0054','\u0055','\u0056','\u0057','\u0058','\u0059','\u005A','\u0030', '\u0031','\u0032','\u0033','\u0034','\u0035','\u0036','\u0037','\u0038','\u0039','\u004F']
# LABEL = ["\u00B7",'\u7696','\u6CAA','\u6D25','\u6E1D','\u5180','\u664B','\u8499','\u8FBD','\u5409',"\u9ED1","\u82CF","\u6D59",'\u4EAC','\u95FD','\u8D63','\u9C81', "\u8C6B","\u9102","\u6E58","\u6FB3","\u6842","\u743C",'\u5DDD','\u8D35','\u4E91','\u85CF','\u9655','\u7518','\u9752','\u5B81','\u65B0','\u8B66','\u5B66','\u0041','\u0042','\u0043','\u0044','\u0045','\u0046','\u0047','\u0048','\u004A','\u004B','\u004C','\u004D','\u004E','\u0050','\u0051','\u0052','\u0053','\u0054','\u0055','\u0056','\u0057','\u0058','\u0059','\u005A','\u0030', '\u0031','\u0032','\u0033','\u0034','\u0035','\u0036','\u0037','\u0038','\u0039','\u004F']
# LABEL = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789·')
# MAX_CHARS = 12
# MAX_LEN = MAX_CHARS + 2

def labelDictionary():
    labels = list(cfg.chardict)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

def label_converter(text):
    text_list = list(cfg.chardict)
    tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
    vector = np.zeros(cfg.MAX_LEN)
    num_classes, letter2index, index2letter = labelDictionary()
    # if len(text) == 6:
    #     text = text.replace('·','-')
    for idx,i in enumerate(text):
        i = i.upper()
        if i == 'I':
            i = '1'
        elif i == 'O':
            i = '0'
        index = letter2index[i] + 3
        vector[idx+1] = index
    vector[len(text)+1] = tokens['END_TOKEN']
    if vector[-1] != tokens['END_TOKEN']:
        vector[len(text)+2:] = tokens['PAD_TOKEN']

    return vector
def label_to_string(label):
    label = label[1:]
    stop_index = np.where(label==1)[0][0]
    plate_label = list(label[:stop_index])
    num_classes, letter2index, index2letter = labelDictionary()
    str = [index2letter[label-3] for label in plate_label]
    plate = ''.join(str)

    return plate

def lmdb_get(img_path,mask=False):
    if mask is not True:
        arr = cv2.cvtColor(cv2.imdecode(np.frombuffer((img_path),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
    else:
        ret, arr = cv2.threshold(cv2.cvtColor(cv2.imdecode(np.frombuffer((img_path),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)
    return arr

class Dataloader_SPG(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir

            env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.nSamples = int(txn.get('num-samples'.encode()))
            self.name_list = []
            for index in range(int(self.nSamples)):
                index += 1  # lmdb starts with 1
                imagename = 'image-%09d' % index
                self.name_list.append(imagename)

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
        
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:

            i_t = txn.get(img_name.replace('image','label').encode()).decode()  
            i_t = label_converter(i_t).astype(int)
            i_s = lmdb_get(txn.get((cfg.i_s_dir +'_' + img_name).encode()))
            t_b = lmdb_get(txn.get((cfg.t_b_dir +'_' + img_name).encode()))
            t_f = lmdb_get(txn.get((cfg.t_f_dir +'_' + img_name).encode()))            
            mask_t = lmdb_get(txn.get((cfg.mask_t_dir +'_' + img_name).encode()),mask=True)
            mask_s = lmdb_get(txn.get((cfg.mask_s_dir +'_' + img_name).encode()),mask=True)

        return [i_t, i_s, t_b, t_f, mask_t, mask_s]

class Dataloader_SPG_combine(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir

            env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.nSamples = int(txn.get('num-samples'.encode()))
            self.source_name_list = []
            self.target_name_list = []
            for index in range(int(self.nSamples)):
                index += 1  # lmdb starts with 1
                s_imagename = 'image2-%09d' % index
                t_imagename = 'image-%09d' % index
                self.source_name_list.append(s_imagename)
                self.target_name_list.append(t_imagename)

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
        
      
    def __len__(self):
        return len(self.target_name_list)
    
    def __getitem__(self, idx):

        s_img_name = self.source_name_list[idx]
        t_img_name = self.target_name_list[idx]

        env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:

            i_ts = txn.get(s_img_name.replace('image2','label2').encode()).decode()
            i_tt = txn.get(t_img_name.replace('image','label').encode()).decode()  
            text = i_tt
            # text = i_tt.replace('·', '')
            i_ts = label_converter(i_ts).astype(int)
            i_tt = label_converter(i_tt).astype(int)
            i_s = lmdb_get(txn.get((cfg.i_s_dir +'_' + t_img_name).encode()))
            t_b = lmdb_get(txn.get((cfg.t_b_dir +'_' + t_img_name).encode()))
            t_f = lmdb_get(txn.get((cfg.t_f_dir +'_' + t_img_name).encode()))            
            mask_t = lmdb_get(txn.get((cfg.mask_t_dir +'_' + t_img_name).encode()),mask=True)
            mask_s = lmdb_get(txn.get((cfg.mask_s_dir +'_' + t_img_name).encode()),mask=True)

        return [i_ts, i_tt, i_s, t_b, t_f, mask_t, mask_s, text]

class dataloader_fintune(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir

            env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.nSamples = int(txn.get('num-samples'.encode()))
            self.name_list = []
            for index in range(int(self.nSamples)):
                index += 1  # lmdb starts with 1
                imagename = 'image-%09d' % index
                self.name_list.append(imagename)

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:

            i_s = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.i_s_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            i_t = txn.get(img_name.replace('image','label').encode()).decode()  
            i_t = label_converter(i_t).astype(int)    
            t_t = txn.get(img_name.replace('image','label2').encode()).decode()  
            t_t = label_converter(t_t).astype(int)         
            t_f = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.t_f_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            

        return [i_t, i_s, t_f,t_t]

class dataloader_fintune_real_ori(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = os.path.join(cfg.select_data_dir,'90')
            self.data_dir_gen = os.path.join(cfg.select_data_dir,'100')
            self.name_list = os.listdir(self.data_dir)

            
            self.label = {}
            with open(os.path.join(cfg.finetune_dir,'source_list2.txt'),'r',encoding='utf-8') as fw:
                data_list = fw.readlines()
                for data in data_list:
                    image_key,label =  data.strip().split(' ')
                    self.label.update({
                        image_key:label
                    })

            self.label2 = {}

            with open(os.path.join(cfg.finetune_dir,'target_list2.txt'),'r',encoding='utf-8') as fw2:
                data_list2 = fw2.readlines()
                for data in data_list2:
                    image_key,label =  data.strip().split(' ')
                    self.label2.update({
                        image_key:label
                    })

            self.train_path = cfg.finetune_dir
            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        if '_2' in img_name:
            i_s = io.imread(os.path.join(self.train_path,cfg.t_f_dir, img_name.replace('_2','')))
            t_f = io.imread(os.path.join(self.data_dir_gen, img_name.replace('_2','')))
        else:
            i_s = io.imread(os.path.join(self.data_dir, img_name))
            t_f = io.imread(os.path.join(self.train_path,cfg.t_f_dir, img_name))
        i_t = self.label[img_name]
        i_t = label_converter(i_t).astype(int)   
        t_t = self.label2[img_name]
        t_t = label_converter(t_t).astype(int)  
    
        
        rotate = np.random.choice([0,90,-90])
        if rotate ==90:
            i_s = cv2.rotate(i_s, cv2.ROTATE_90_CLOCKWISE)
            t_f = cv2.rotate(t_f, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == -90:
            i_s = cv2.rotate(i_s, cv2.ROTATE_90_COUNTERCLOCKWISE)
            t_f = cv2.rotate(t_f, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return [i_t, i_s, t_f,t_t]
    
class dataloader_fintune_real(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.root = cfg.finetune_dir
            self.name_list = []
                        
            self.target_label = {}
            with open(os.path.join(self.root,'source_list.txt'),'r',encoding='utf-8') as fw:
                data_list = fw.readlines()
                for data in data_list:
                    image_key,label =  data.strip().split(' ')
                    self.name_list.append(image_key)
                    self.target_label.update({
                        image_key:label
                    })

            self.source_label = {}

            with open(os.path.join(self.root,'target_list.txt'),'r',encoding='utf-8') as fw2:
                data_list2 = fw2.readlines()
                for data in data_list2:
                    image_key,label =  data.strip().split(' ')
                    self.source_label.update({
                        image_key:label
                    })

            # self.label3 = {}

            # with open(os.path.join(cfg.finetune_dir,'target_list3.txt'),'r',encoding='utf-8') as fw3:
            #     data_list3 = fw3.readlines()
            #     if len(data_list3) != 0:
            #         for data in data_list3:
            #             image_key,label =  data.strip().split(' ')
            #             new_image_key = image_key.replace('.','_2.')
            #             self.label.update({
            #                 new_image_key:label
            #             })
            #             self.label2.update({
            #                 new_image_key:self.label[image_key]
            #             })
            #             self.name_list.append(new_image_key)
                        

            self.train_path = cfg.finetune_dir
            


            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        
        i_s = io.imread(os.path.join(self.root,cfg.t_f_dir, img_name))
        t_f = io.imread(os.path.join(self.train_path,cfg.i_s_dir, img_name))
        i_t = self.target_label[img_name]
        i_t = label_converter(i_t).astype(int)   
        t_t = self.source_label[img_name]
        t_t = label_converter(t_t).astype(int)  
    
        return [i_t, i_s, t_f,t_t]
    
class dataloader_fintune_real_combine(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.root = cfg.finetune_dir
            self.name_list = []
                        
            self.target_label = {}
            with open(os.path.join(self.root,'gt.txt'),'r',encoding='utf-8') as fw:
                data_list = fw.readlines()
                for data in data_list:
                    image_key,label =  data.strip().split(' ')
                    self.name_list.append(image_key)
                    self.target_label.update({
                        image_key:label
                    })

            self.source_label = {}

            with open(os.path.join(self.root,'gt2.txt'),'r',encoding='utf-8') as fw2:
                data_list2 = fw2.readlines()
                for data in data_list2:
                    image_key,label =  data.strip().split(' ')
                    self.source_label.update({
                        image_key:label
                    })

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        i_s = io.imread(os.path.join(self.root,cfg.i_s_dir, img_name))
        t_f = io.imread(os.path.join(self.root,cfg.t_f_dir, img_name))
        
        ### i_ts : target_label ### 
        ### i_tt : source_label ### 

        i_ts = self.target_label[img_name]
        i_ts = label_converter(i_ts).astype(int)   
        i_tt = self.source_label[img_name]
        i_tt = label_converter(i_tt).astype(int)  
    
        return [i_ts, i_s, t_f, i_tt]
    
class test_dataset_finetune(Dataset):
    
    def __init__(self, cfg, transform = None):
        
        self.root = cfg.finetune_dir
        self.label_dict = {}
        self.img_list = []

        # with open(os.path.join(self.root, 'source_list.txt'),'r',encoding='utf-8') as f:
        #     data_list = f.readlines()

        #     char_dict = {}

        #     for i in data_list:
        #         image,label = i.strip('\n').split(' ')
        #         char_dict.update({
        #         image:label
        #         })

        # with open('/home/avlab/deep-text-recognition-benchmark-master/lp2022/LP2022_MIX_Train/target_list2.txt','w',encoding='utf-8')as ff:
        #     for img_name,label in char_dict.items():
        #         if len(label) == 8:
        #             gen_label = gen_plate_text.random_plate(7)
        #         elif len(label) == 7:
        #             gen_label = gen_plate_text.random_plate(6)
        #             label_point_index = list(label).index('·')
        #             new_label_point_index = list(gen_label).index('·')
        #             while new_label_point_index != label_point_index:
        #                 gen_label = gen_plate_text.random_plate(6)
        #                 new_label_point_index = list(gen_label).index('·')
        #         ff.writelines(img_name+' '+gen_label+'\n')

        with open (os.path.join(self.root,'target_list.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                name, label = i.strip('\n').split(' ')
                self.img_list.append(name)
                self.label_dict.update({
                    name:label
                })

        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_name = self.img_list[idx]
        # len_i_t = len(self.label_dict[img_name])
        i_t = self.label_dict[img_name]
        # i_t = gen_plate_text.random_plate(len_i_t-1)
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(self.root,cfg.i_s_dir, img_name))

        sample = (i_t, i_s, img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class test_dataset_finetune_combine_self(Dataset):
    
    def __init__(self, cfg, transform = None):
        
        self.root = cfg.finetune_dir
        self.s_label_dict = {}
        self.s_img_list = []
        self.t_label_dict = {}
        self.t_img_list = []

        with open (os.path.join(self.root,'gt.txt'),'r',encoding='utf-8') as f:
            self.s_text_list = f.readlines()
            for i in self.s_text_list:
                name, label = i.strip('\n').split(' ')
                self.s_img_list.append(name)
                self.s_label_dict.update({
                    name:label
                })

        with open (os.path.join(self.root,'gt2.txt'),'r',encoding='utf-8') as f:
            self.t_text_list = f.readlines()
            for i in self.t_text_list:
                name, label = i.strip('\n').split(' ')
                self.t_img_list.append(name)
                self.t_label_dict.update({
                    name:label
                })

        
    def __len__(self):
        return len(self.s_img_list)
    
    def __getitem__(self, idx):
        
        s_img_name = self.s_img_list[idx]
        t_img_name = self.t_img_list[idx]

        i_ts = self.s_label_dict[s_img_name]
        i_tt = self.t_label_dict[t_img_name]

        i_ts = label_converter(i_ts).astype(int)
        i_tt = label_converter(i_tt).astype(int)

        i_s = io.imread(os.path.join(self.root,cfg.t_f_dir, t_img_name))

        sample = (i_ts, i_tt, i_s, t_img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class dataloader_fintune_cycle(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir

            env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.nSamples = int(txn.get('num-samples'.encode()))
            self.name_list = []
            for index in range(int(self.nSamples)):
                index += 1  # lmdb starts with 1
                imagename = 'image-%09d' % index
                self.name_list.append(imagename)

            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        env = lmdb.open(self.data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:

            i_s = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.i_s_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
            i_t = txn.get(img_name.replace('image','label').encode()).decode()  
            i_t = label_converter(i_t).astype(int)    
            t_t = txn.get(img_name.replace('image','label2').encode()).decode()  
            t_t = label_converter(t_t).astype(int)         
            t_f = cv2.cvtColor(cv2.imdecode(np.frombuffer((txn.get((cfg.t_f_dir +'_' + img_name).encode())),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)

        return [i_t, i_s, t_f,t_t]

class test_dataset(Dataset):
    
    def __init__(self, cfg, transform = None):
        
        self.files = os.listdir(cfg.test_data_dir)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.transform = transform
        self.t_b_dir = cfg.i_s_dir
        self.name_list = os.listdir(os.path.join(cfg.test_data_dir, self.t_b_dir))
        with open (os.path.join(cfg.test_data_dir,'target_list.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
        self.img_list = [i.split(' ')[0] for i in self.text_list]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        index = self.img_list.index(img_name)
        i_t = self.text_list[index].split(' ')[1].strip('\n')
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(cfg.test_data_dir,cfg.i_s_dir, img_name))
        

        sample = (i_t, i_s, img_name)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class test_dataset2(Dataset):
    
    def __init__(self, cfg, transform = None):
        
        self.files = os.listdir(cfg.test2_data_dir)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.label_dict = {}
        self.img_list = []

        with open('/home/avlab/deep-text-recognition-benchmark-master/lp2022/LP2022_MIX_Train/source_list.txt','r',encoding='utf-8') as f:
            data_list = f.readlines()

            char_dict = {}

            for i in data_list:
                image,label = i.strip('\n').split(' ')
                char_dict.update({
                image:label
                })

        with open('/home/avlab/deep-text-recognition-benchmark-master/lp2022/LP2022_MIX_Train/target_list2.txt','w',encoding='utf-8')as ff:
            for img_name,label in char_dict.items():
                if len(label) == 8:
                    gen_label = gen_plate_text.random_plate(7)
                elif len(label) == 7:
                    gen_label = gen_plate_text.random_plate(6)
                    label_point_index = list(label).index('·')
                    new_label_point_index = list(gen_label).index('·')
                    while new_label_point_index != label_point_index:
                        gen_label = gen_plate_text.random_plate(6)
                        new_label_point_index = list(gen_label).index('·')
                ff.writelines(img_name+' '+gen_label+'\n')

        with open (os.path.join(cfg.test2_data_dir,'target_list2.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                name, label = i.strip('\n').split(' ')
                self.img_list.append(name)
                self.label_dict.update({
                    name:label
                })

        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_name = self.img_list[idx]
        # len_i_t = len(self.label_dict[img_name])
        i_t = self.label_dict[img_name]
        # i_t = gen_plate_text.random_plate(len_i_t-1)
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(cfg.test2_data_dir,cfg.i_s_dir, img_name))

        sample = (i_t, i_s, img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class test_dataset_synth(Dataset):
    
    def __init__(self, cfg, transform = None):


        self.root = cfg.test_data_dir
        self.files = os.listdir(self.root)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.label_dict = {}
        self.img_list = []
        with open (os.path.join(self.root,'gt2.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                name, label = i.strip('\n').split(' ')
                self.img_list.append(name)
                self.label_dict.update({
                    name:label
                })

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_name = self.img_list[idx]
        i_t = self.label_dict[img_name]
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(self.root,cfg.i_s_dir, img_name))

        sample = (i_t, i_s, img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class test_dataset_synth_combine(Dataset):
    
    def __init__(self, cfg, transform = None):
        self.root = cfg.test_data_dir
        self.files = os.listdir(self.root)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.s_label_dict = {}
        self.s_img_list = []
        self.t_label_dict = {}
        self.t_img_list = []
        with open (os.path.join(self.root,'gt.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                name, label = i.strip('\n').split(' ')
                self.s_img_list.append(name)
                self.s_label_dict.update({
                    name:label
                })
        with open (os.path.join(self.root,'gt2.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                name, label = i.strip('\n').split(' ')
                self.t_img_list.append(name)
                self.t_label_dict.update({
                    name:label
                })

    def __len__(self):
        return len(self.t_img_list)
    
    def __getitem__(self, idx):
        
        s_img_name = self.s_img_list[idx]
        t_img_name = self.t_img_list[idx]
        i_ts = self.s_label_dict[s_img_name]
        i_tt = self.t_label_dict[t_img_name]
        i_ts = label_converter(i_ts).astype(int)
        i_tt = label_converter(i_tt).astype(int)

        i_s = io.imread(os.path.join(self.root,cfg.i_s_dir, t_img_name))

        sample = (i_ts, i_tt, i_s, t_img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample

class predict_dataset(Dataset):
    
    def __init__(self, input_dir, transform = None):
        
        self.input_dir = input_dir
        self.files = os.listdir(input_dir)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        # self.transform = transform
        self.i_s_dir = cfg.i_s_dir
        self.img_list = []
        self.label_list = {}
        with open (os.path.join(input_dir,'gt2.txt'),'r',encoding='utf-8') as f:
            self.text_list = f.readlines()
            for i in self.text_list:
                image_name , label = i.split(' ')
                self.img_list.append(image_name)
                self.label_list.update({
                    image_name:label.strip()
                })
        self.img_list = [i.split(' ')[0] for i in self.text_list]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_name = self.img_list[idx]
        # img_name_1 = str(int(img_name.split('.')[0])).zfill(5)+'.png
        
        i_t =self.label_list[img_name]
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(self.input_dir,cfg.i_s_dir, img_name))
        
        

        sample = (i_t, i_s, img_name)
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample



class dataloader_predict_use_syn_cheack_real(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.root = cfg.finetune_dir
            self.name_list = []
                        
            self.target_label = {}
            with open(os.path.join(self.root,'source_list.txt'),'r',encoding='utf-8') as fw:
                data_list = fw.readlines()
                for data in data_list:
                    image_key,label =  data.strip().split(' ')
                    self.name_list.append(image_key)
                    self.target_label.update({
                        image_key:label
                    })

            self.source_label = {}

            with open(os.path.join(self.root,'target_list.txt'),'r',encoding='utf-8') as fw2:
                data_list2 = fw2.readlines()
                for data in data_list2:
                    image_key,label =  data.strip().split(' ')
                    self.source_label.update({
                        image_key:label
                    })

            # self.label3 = {}

            # with open(os.path.join(cfg.finetune_dir,'target_list3.txt'),'r',encoding='utf-8') as fw3:
            #     data_list3 = fw3.readlines()
            #     if len(data_list3) != 0:
            #         for data in data_list3:
            #             image_key,label =  data.strip().split(' ')
            #             new_image_key = image_key.replace('.','_2.')
            #             self.label.update({
            #                 new_image_key:label
            #             })
            #             self.label2.update({
            #                 new_image_key:self.label[image_key]
            #             })
            #             self.name_list.append(new_image_key)
                        

            self.train_path = cfg.finetune_dir
            


            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        
        i_s = io.imread(os.path.join(self.root,cfg.t_f_dir, img_name))
        t_f = io.imread(os.path.join(self.train_path,cfg.i_s_dir, img_name))
        i_t = self.target_label[img_name]
        i_t = label_converter(i_t).astype(int)   
        t_t = self.source_label[img_name]
        t_t = label_converter(t_t).astype(int)  
    
        return [i_t, i_s, t_f,t_t]


class demo_dataset(Dataset):
    
    def __init__(self, input_path,input_text, transform = None):
        
            if os.path.isdir(input_path):
                self.path = input_path   
                self.name_list = []
                self.text_list = {}
                if os.path.isfile(input_text):
                    with open (input_text,'r',encoding='utf-8') as f:
                        data = f.readlines()
                        for i in data:
                            image_name,label = i.strip('\n').split(' ')
                            self.name_list.append(image_name)
                            self.text_list.update({
                                image_name:label
                        })
            else:
                self.path = input_path.strip(input_path.split('/')[-1])
                self.name_list = [input_path.split('/')[-1]]
                self.text_list = {str (input_path.split('/')[-1]):str(input_text)}
                    


    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        # img_name_1 = str(int(img_name.split('.')[0])).zfill(5)+'.png'
        
        i_t =  self.text_list[img_name]
        i_t = label_converter(i_t).astype(int)

        i_s = io.imread(os.path.join(self.path, img_name))
        
        

        sample = (i_t, i_s, img_name)
        
        return sample
    
class demo_dataset_combine(Dataset):
    
    def __init__(self, input_path,input_text, transform = None):
        
            if os.path.isdir(input_path):
                self.path = input_path   
                self.source_name_list = []
                self.source_text_list = {}
                self.target_name_list = []
                self.target_text_list = {}
                if os.path.isdir(input_text):
                    with open (os.path.join(input_text,'gt.txt'),'r',encoding='utf-8') as f:
                        data = f.readlines()
                        for i in data:
                            image_name,label = i.strip('\n').split(' ', 1)
                            self.source_name_list.append(image_name)
                            self.source_text_list.update({
                                image_name:label
                        })
                    with open (os.path.join(input_text,'gt2.txt'),'r',encoding='utf-8') as f:
                        data = f.readlines()
                        for i in data:
                            image_name,label = i.strip('\n').split(' ', 1)
                            self.target_name_list.append(image_name)
                            self.target_text_list.update({
                                image_name:label
                        })
            else:
                self.path = input_path.strip(input_path.split('/')[-1])
                self.name_list = [input_path.split('/')[-1]]
                self.text_list = {str (input_path.split('/')[-1]):str(input_text)}
                    


    def __len__(self):
        return len(self.source_name_list)
    
    def __getitem__(self, idx):
        
        # img_name = self.name_list[idx]
        s_img_name = self.source_name_list[idx]
        t_img_name = self.target_name_list[idx]
        # img_name_1 = str(int(img_name.split('.')[0])).zfill(5)+'.png'
        i_ts =  self.source_text_list[s_img_name]
        i_ts = label_converter(i_ts).astype(int)
        i_tt =  self.target_text_list[t_img_name]
        i_tt = label_converter(i_tt).astype(int)

        i_s = io.imread(os.path.join(self.path, t_img_name))
        
        

        sample = (i_ts, i_tt, i_s, t_img_name)
        
        return sample

class datagen_srnet(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir
            self.t_b_dir = cfg.t_b_dir
            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
            self.name_list = os.listdir(os.path.join(self.data_dir, self.t_b_dir))
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        
        i_t = io.imread(os.path.join(cfg.data_dir, cfg.i_t_dir, img_name))
        i_s = io.imread(os.path.join(cfg.data_dir, cfg.i_s_dir, img_name))
        t_sk = io.imread(os.path.join(cfg.data_dir, cfg.t_sk_dir, img_name), as_gray = True)
        t_t = io.imread(os.path.join(cfg.data_dir, cfg.t_t_dir, img_name))
        t_b = io.imread(os.path.join(cfg.data_dir, cfg.t_b_dir, img_name))
        t_f = io.imread(os.path.join(cfg.data_dir, cfg.t_f_dir, img_name))
        mask_t = io.imread(os.path.join(cfg.data_dir, cfg.mask_t_dir, img_name), as_gray = True)
        
        return [i_t, i_s, t_sk, t_t, t_b, t_f, mask_t]
    

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
        
class example_dataset(Dataset):
    
    def __init__(self, data_dir = cfg.test_data_dir, transform = None):
        
        self.files = os.listdir(data_dir)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.transform = transform
        self.t_b_dir = cfg.i_s_dir
        self.name_list = os.listdir(os.path.join(cfg.test_data_dir, self.t_b_dir))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        img_name = self.files[idx]
        
        i_t = io.imread(os.path.join(cfg.example_data_dir, img_name + 'i_t.png'))
        i_s = io.imread(os.path.join(cfg.example_data_dir, img_name + 'i_s.png'))
        
        sample = (i_t, i_s, img_name )
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    
        
class To_tensor(object):
    def __call__(self, sample):
        
        i_t, i_s, img_name, padding_list,img_shape = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)

        return (i_t.float(), i_s.float(), img_name, padding_list,img_shape)

class To_tensor_label(object):
    def __call__(self, sample):
        
        i_t, i_s, img_name, padding_list,img_shape,mask_t = sample
        
        # i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1
        mask_t = mask_t.transpose((2, 0, 1))/255.

        i_t = torch.from_numpy(i_t.astype(int))
        i_s = torch.from_numpy(i_s)
        mask_t = torch.from_numpy(mask_t)

        return (i_t, i_s.float(), img_name, padding_list,img_shape,mask_t)
        
    
