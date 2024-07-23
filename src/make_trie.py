import json
import numpy as np
import torch
import transformers
from transformers import T5Tokenizer, T5TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pickle
import pandas as pd

from src.dataset import get_oldid2newid


def get_trie(args,tokenizer):
    """
    return trie structure
    """
    oldid2newid = get_oldid2newid(args)
    id_list = get_id_list(oldid2newid)
    id_trie = tokenize_and_make_trie(id_list,tokenizer)
    return id_trie

def get_id_list(oldid2newid:dict):
    
    print("ID sample: ",list(oldid2newid.values())[:5])
    
    id_list = []
    for line in oldid2newid.values():
        id_list.append(str(line))
    
    return id_list

def tokenize_and_make_trie(id_list,tokenizer:AutoTokenizer):
    trie = Trie()
    
    special_tokens = tokenizer.special_tokens_map
    print(special_tokens)
    
    eos = 50277 #tokenizer.convert_tokens_to_ids("<|endoftext|>")
    print("eos: ",eos)
    flag=0
    for id in tqdm(id_list):
                
        tokenID1 = [50277] + tokenizer(id)['input_ids'] + [eos] #convert_tokens_to_ids
        tokenID2 = [5417] + tokenizer(id)['input_ids'] + [eos] #convert_tokens_to_ids
        #209,  4118,  5417

        if flag%10000==0:
            print(id, tokenizer(id)['input_ids'])
            print(tokenizer.decode(tokenID1))
            print(tokenizer.decode(tokenID2))
        
        trie.insert(tokenID1)
        trie.insert(tokenID2)
        flag+=1
    
    return trie
    
        
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.children.keys()




def main():
    ex=None
    id_list = get_id_list()
    
    tokenizer = AutoTokenizer.from_pretrained('anas-awadalla/mpt-1b-redpajama-200b')
    mytrie= tokenize_and_make_trie(id_list,tokenizer)

    #listed_id = list(id_list[1])

    #print(listed_id)

    tokenized = tokenizer(['0','1','2','3','4','5','6','7','8','9'])
    print(tokenizer.tokenize('0'))
    
    print(tokenized)

    print(list(mytrie.search([]))) #11066 11 19729
    
    #print(ex)
    #out = tokenizer.decode(ex)
    #print(out)


if __name__=="__main__":
    main()