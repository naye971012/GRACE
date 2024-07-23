from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch

def get_model(model):
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    return model

class FlickrTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        
    def compute_loss(self, model, inputs, return_outputs=False):
                
        loss = model.forward(vision_x = inputs['image'],
                             lang_x = inputs['input_ids'],
                             attention_mask = inputs['attention_mask'],
                             labels = inputs['imageID']).loss
                             #clear_conditioned_layers = False).loss  #label
                            #clear_conditioned_layers: bool = True,
                            #past_key_values=None,
                            #use_cache: bool = False,)
        
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ): #-> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        model.eval()

        if inputs['image']!=None:
            inputs['image'] = inputs['image'].half() #HALF TENSOR
        
        with torch.no_grad():
            #model = get_model(model)
            batch_beams = model.generate(
                vision_x = inputs['image'],
                lang_x = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                max_length=50,
                num_beams=10, #recall@10
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=10,
                early_stopping=True).reshape(inputs['input_ids'].shape[0], 10, -1) #[32,10,40] #[batch,beam,length]
        
        padding = (0, 50 - batch_beams.shape[2]) #last dim padding
        if padding[1]>0:
            batch_beams = torch.nn.functional.pad(batch_beams,padding, mode='constant', value=0)
        
        return (None, batch_beams, inputs['imageID']) #label = batch,6