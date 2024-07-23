from typing import Tuple
from huggingface_hub import hf_hub_download
import torch
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from safetensors.torch import load_file

from src.open_flamingo import create_model_and_transforms

def get_model(args) -> Tuple:
    """
    model, image_processor, tokenizer order
    """
    
    if args.model_type == "open_flamingo":
        model, image_processor, tokenizer = create_model_and_transforms(
            args,
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=2,
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        
        if args.finetune_model_path==None:
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
            model = model.half() #half precision
        else:
            print("load finetuned model...", args.finetune_model_path)
            state_dict = load_file(args.finetune_model_path)
            model.load_state_dict(state_dict)
            model = model.half() #half precision
        
        #estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)
    
    return (model, image_processor, tokenizer)





if __name__=="__main__":
    class args:
        model_type = "open_flamingo"
        finetune_model_path = "./model/GRACE_Reproduce_step1/checkpoint-33000/model.safetensors"
    get_model(args)