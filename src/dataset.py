from datasets import load_dataset
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from PIL import Image
import io
from typing import Tuple
import torch.nn.functional as F
import json

from src.testdataset import TestDataset_P1, TestDataset_P2


def get_rawdata(args):
    
    if args.dataset_name == "flickr30k":
        raw_ds = load_dataset("nlphuji/flickr30k",cache_dir="./data")
    else:
        raise ValueError(f"dataset name {args.dataset_name} not found")
    return raw_ds

def get_oldid2newid(args):
    """
    get imageID
    """
    with open(f"./data/ID_{args.dataset_name}_{args.IDtype}.json", 'r') as reader:
        oldid2newid = json.load(reader)

    return oldid2newid


def get_dataset(args, image_processor, tokenizer) -> Tuple:
    
    raw_ds = get_rawdata(args)
    
    oldid2newid = get_oldid2newid(args)
    
    train_ds = TrainDataset(args, oldid2newid, raw_ds, image_processor, tokenizer, "train")
    valid_ds = TestDataset_P2(args, oldid2newid, raw_ds, image_processor, tokenizer, "val")
    test_ds = TestDataset_P2(args, oldid2newid, raw_ds, image_processor, tokenizer, "test")
    data_collactor = DataCollator(args, tokenizer=tokenizer, padding=True, max_length=args.max_length)
    
    return train_ds, valid_ds, test_ds, data_collactor



class TrainDataset(Dataset):
    def __init__(
            self,
            args,
            oldid2newid,
            raw_ds,
            image_processor,
            tokenizer,
            split = "train" #train/valid/test
    ):

        self.args = args
        self.split = split
        self.oldid2newid = oldid2newid
        
        #=========== load dataset and use only train_type model ============#
        #['image', 'caption', 'sentids', 'split', 'img_id', 'filename']
        ds = raw_ds['test'].to_pandas()
        ds_group = ds.groupby('split')
        self.ds = ds_group.get_group(split) #train / valid / test
        #==================================================================#
        
        
        #============== Image -> ID task or 5 Annotation -> ID task =========#
        if self.args.train_type == "Learning2Memorize":
            self.ds = ds #because image -> ID training is for all data, including valid/test
            self.ds_all = ds
            self.total_len = len(self.ds)
        elif self.args.train_type == "Learning2Retrieve":
            self.ds_all = ds
            self.total_len = len(self.ds) * 5
        elif self.args.train_type == "JointLearning":
            self.ds_all = ds
            self.total_len = len(self.ds) * 5 + len(self.ds_all) * 2 #5:2
        else:
            raise ValueError(f"{self.args.train_type} is not supported")
        #=====================================================================#
        
        self.data_len = len(self.ds)
        
        #========== tokenizer & image processor ================================#
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        #======================================================================#
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        data = self.ds.iloc[idx%self.data_len]
        
        caption = data['caption'] #5 string of List
        imageID = str(self.oldid2newid[str(data['img_id'])]) #int -> stringID
        #captionID = data['sentids'] #5 int of list
        
        """
        Step 1: Make Caption
        """
        
        """
        Step 2: Preprocessing images
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
        """
        image = Image.open(io.BytesIO(data['image']['bytes']))
        vision_x = self.image_processor(image)
        
        """
        Step 3: Preprocessing text
        Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
        """
        if self.args.train_type == "Learning2Memorize":
            lang_x = self.tokenizer(
                [f"Predict the identifier for the <image> <|endofchunk|>"],
                return_tensors="pt",
                truncation='only_first',
                padding="do_not_pad"
            )
            use_image = True
        elif self.args.train_type == "Learning2Retrieve":
            query = caption[idx//self.data_len] #5 random query
            lang_x = self.tokenizer(
                [f"Predict the image identifier corresponding to the given query: {query} ### ID"],
                return_tensors="pt",
                truncation='only_first',
                padding="do_not_pad"
            )
            use_image = False
        elif self.args.train_type == "JointLearning":
            
            if idx >= self.data_len * 5: #image2id
                
                data = self.ds_all.iloc[idx%len(self.ds_all)] #warning include test query
                image = Image.open(io.BytesIO(data['image']['bytes']))
                vision_x = self.image_processor(image)
                
                lang_x = self.tokenizer(
                    [f"Predict the identifier for the <image> <|endofchunk|>"],
                    return_tensors="pt",
                    truncation='only_first',
                    padding="do_not_pad"
                )
                use_image = True
            else: #query2id
                query = caption[idx//self.data_len] #5 random query
                lang_x = self.tokenizer(
                    [f"Predict the image identifier corresponding to the given query: {query} ### ID"],
                    return_tensors="pt",
                    truncation='only_first',
                    padding="do_not_pad"
                )
                use_image = False
            
        else:
            raise ValueError(f"{self.args.train_type} is not supported")
        
        output = {
            "query" : lang_x,
            "image": vision_x,
            "imageID": imageID,
            "split": self.split,
            "use_image": use_image
        }
        
        return output


#right padding
def pad_tensor(tensor, max_length, value):
    return F.pad(tensor, (0, max_length - len(tensor)), mode='constant', value=value)

#left madding
def pad_tensor_left(tensor, max_length, value):
    return F.pad(tensor, (max_length - len(tensor), 0), mode='constant', value=value)


@dataclass
class DataCollator(DataCollatorWithPadding):
    def __init__(self,args,**kwargs):
        super().__init__(**kwargs)
        self.args = args

    def __call__(self, features):
        """
        since flamingo is decoder-only model, input format shuold be following...
        inputs_ids: [query + label] (teacher forcing)
        labels: [-100 * query_len, label]
        """
    
        #=========== same as encoder-decoder collactor ==========================#
        input_ids = [item['query']['input_ids'] for item in features] #[b,1, len]
        attention_mask = [item['query']['attention_mask'] for item in features] #[b,1, len]
        images = [item['image'] for item in features] #[b,c,w,h]
        image_ids = [item['imageID'] for item in features] #[id]...
        
        images = torch.stack(images).unsqueeze(1).unsqueeze(1)  # Shape: [b, 1, 1, c, w, h]
    
        if self.args.IDtype == "stringID": #0~31000
            tokenized_image_ids = [self.tokenizer(image_id, padding='do_not_pad', return_tensors='pt') for image_id in image_ids]
            labels = [pad_tensor(t['input_ids'][0], 6, self.tokenizer.pad_token_id) for t in tokenized_image_ids]
        else:
            raise ValueError(f"IDtype {self.args.IDtype} not defined")        
        #======================================================================#
        
        #when constrained generation
        if features[0].get('split') != "train":
            
            #======= pad and stack ================#
            input_ids = [pad_tensor_left(t[0], 30,self.tokenizer.pad_token_id) for t in input_ids]
            attention_mask = [pad_tensor_left(t[0], 30,0) for t in attention_mask]
            
            input_ids = torch.stack(input_ids) #[b,len]
            attention_mask = torch.stack(attention_mask).squeeze(1) #[b,1len]
            labels = torch.stack(labels) #[batch, 6]
            labels[labels == self.tokenizer.pad_token_id] = -100 #for training
            #======================================#

            images = torch.zeros_like(images) #do not use image
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image': images,
                'imageID': labels 
            }
        
        #when teacher-forcing training
        else:
            #====== [batch, query+id+padded] ============================================#
            query_id_concat = [torch.cat([a[0],b]) for a,b in zip(input_ids, labels)] #[b, len + 10]
            query_id_concat = [pad_tensor(t, self.args.max_length,self.tokenizer.pad_token_id) for t in query_id_concat] #[b,len + 10]
            query_id_concat = torch.stack(query_id_concat) #[b,len+10]

            label_concat = query_id_concat.clone()
            query_id_atten = torch.zeros_like(label_concat, dtype=torch.long) #[b,len+label_len] 
            for i, (query, label) in enumerate(zip(input_ids, labels)): #len for each iter
                query_length = len(query[0])
                label_length = len(label)
                
                query_id_atten[i, :query_length + label_length] = 1
                label_concat[i, :query_length-1] = self.tokenizer.pad_token_id
            
            label_concat[label_concat == self.tokenizer.pad_token_id] = -100 #for training
            #============================================================================#
            
            if self.args.train_type == "Learning2Retrieve" or not features[0]['use_image']:
                images = torch.zeros_like(images)

            return {
                'input_ids': query_id_concat,
                'attention_mask': query_id_atten,
                'image': images,
                'imageID': label_concat 
            }