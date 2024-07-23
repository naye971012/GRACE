import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback, AutoModel, AutoTokenizer, T5EncoderModel
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

from src.trainer import FlickrTrainer
from src.model import get_model
from src.dataset import get_dataset
from src.make_trie import get_trie
from src.utils import extract_number

def seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed(42)

tokenizer = None #global
def compute_metrics(eval_preds):
    num_data = 0
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    
    
    predictions_list = [eval_preds.predictions[i] for i in range(eval_preds.predictions.shape[0])]
    labels_list = [eval_preds.label_ids[i] for i in range(eval_preds.label_ids.shape[0])]

    for i, (predict, label) in enumerate(zip(predictions_list, labels_list)): #[batch, 10, 20] and [batch, 20]
        num_data += 1
                        
        label = [l if l!=-100 else 0 for l in label]

        pred_decode = tokenizer.batch_decode(predict, skip_special_tokens=True) #list[str]
        label_decode = tokenizer.batch_decode([label], skip_special_tokens=True)[0] #str

        pred_decode = [extract_number(phrase) for phrase in pred_decode] #only get id
        if(i%100==0):
            print(label_decode)
            print(pred_decode)
        
        if label_decode in pred_decode:
            recall_10+=1
            if label_decode in pred_decode[:5]:
                recall_5+=1
            if label_decode==pred_decode[0]:
                recall_1+=1
        
    return_dict = dict()
    return_dict.update({'R@1': recall_1/num_data,
                        'R@5': recall_5/num_data,
                        'R@10': recall_10/num_data})

    return return_dict


def main(args):
    global tokenizer
    
    print(f"loading model...", {args.model_type})
    model, image_processor, tokenizer = get_model(args)

    print(f"loading dataset...", {args.dataset_name}, {args.train_type})
    #do not use test_ds in training
    train_dataset, valid_dataset, _, data_collactor = get_dataset(args, image_processor, tokenizer)

    print(f"load ID trie...")
    id_trie = get_trie(args,tokenizer)
 
    def restric_decode_vocab(batch_idx, prefix_beam):
        prefix_beam = prefix_beam.cpu().numpy()[30-1:].tolist()
        next = id_trie.search(prefix_beam) #trie after max_length index
        if type(next)!=bool and len(list(next))!=0:
            next = list(next)
        else:
            next = [tokenizer.eos_token_id]
            
        assert len(next)!=0, f"next vocab should be nonzero, {next}"

        return next
    
    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    #wandb.login()
    #wandb.init(project="Multimoal_GenIR", name='GRACE_reproduce')
    
    training_args = TrainingArguments(
        output_dir="./model/GRACE_Reproduce_step1",
        learning_rate=args.lr,
        warmup_steps=args.train_steps//10,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        max_steps=args.train_steps, 
        dataloader_drop_last=False,
        evaluation_strategy='steps',
        eval_steps=1500, 
        save_steps=1500,
        logging_steps=50, 
        save_total_limit=2, 
        dataloader_num_workers=8,
        remove_unused_columns=False, #essential!!,
        #deepspeed="./scripts/deepspeed_config.json"
    )

    trainer = FlickrTrainer(
        restrict_decode_vocab=restric_decode_vocab, #constrained decoding
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator= data_collactor
    )
    
    trainer.evaluate() 



if __name__=="__main__":

    class args:
        model_type = "open_flamingo"
        dataset_name = "flickr30k"
        train_type = "Learning2Memorize"
        lr = 1e-4
        train_steps = 100000
        batch_size = 32 #64
        max_length = 20
        IDtype = "stringID"
        grad_accumulation = 1
        finetune_model_path = "./model/GRACE_Reproduce_step1/checkpoint-81000/model.safetensors"

    main(args)