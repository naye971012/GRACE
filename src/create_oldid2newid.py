import json
from datasets import load_dataset

def get_rawdata(args):
    
    #['image', 'caption', 'sentids', 'split', 'img_id', 'filename']
    if args.dataset_name == "flickr30k":
        raw_ds = load_dataset("nlphuji/flickr30k",cache_dir="./data")
    else:
        raise ValueError(f"dataset name {args.dataset_name} not found")
    return raw_ds



def make_string_id(raw_ds,args):
        
    oldid2newid = dict()
    ds = raw_ds['test'].to_pandas()
    
    for i in range(len(ds)):
        cur = ds.iloc[i]
        
        oldid = cur['img_id']
        newid = i+1
        oldid2newid[str(oldid)] = str(newid)

    with open(args.save_dir, 'w') as reader:
        json.dump(oldid2newid, reader, indent=4)        





if __name__=="__main__":

    class args:
        dataset_name = "flickr30k"
        IDtype = "stringID"
        save_dir = f"./data/ID_{dataset_name}_{IDtype}.json"
            
    raw_ds = get_rawdata(args)
    
    if args.IDtype =="stringID":
        make_string_id(raw_ds,args)