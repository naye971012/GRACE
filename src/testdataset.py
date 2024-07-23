from torch.utils.data import Dataset
from PIL import Image
import io
import torch

class TestDataset_P1(Dataset):
    def __init__(
            self,
            args,
            oldid2newid,
            raw_ds,
            image_processor,
            tokenizer,
            split = "test"
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

        self.total_len = len(self.ds)
    
        #========== tokenizer & image processor ================================#
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        #======================================================================#
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        data = self.ds.iloc[idx]
        
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
        lang_x = self.tokenizer(
                [f"Predict the identifier for the <image> <|endofchunk|>"],
                return_tensors="pt",
                truncation='only_first',
                padding="do_not_pad"

        )

        output = {
            "query" : lang_x,
            "image": vision_x,
            "imageID": imageID,
            "split": self.split,
        }
        
        return output


class TestDataset_P2(Dataset):
    def __init__(
            self,
            args,
            oldid2newid,
            raw_ds,
            image_processor,
            tokenizer,
            split = "test"
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

        self.total_len = len(self.ds) * 5
        self.data_len = len(self.ds)

        #========== tokenizer & image processor ================================#
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        #======================================================================#
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        if idx >= self.data_len:
            idx %= self.data_len
        
        data = self.ds.iloc[idx]
        
        caption = data['caption'] #5 string of List
        imageID = str(self.oldid2newid[str(data['img_id'])]) #int -> stringID
        #captionID = data['sentids'] #5 int of list
        

        image = Image.open(io.BytesIO(data['image']['bytes']))
        vision_x = self.image_processor(image)
        

        query = caption[idx%5] #5 random query
        lang_x = self.tokenizer(
                [f"Predict the image identifier corresponding to the given query: {query} ### ID"],
                return_tensors="pt",
                truncation='only_first',
                padding="do_not_pad"
            )
        
        output = {
            "query" : lang_x,
            "image": vision_x,
            "imageID": imageID,
            "split": self.split,
        }
        
        return output
