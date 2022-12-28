import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml
import random
import pathlib,os
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    '''
    Seeds everything so as to allow for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text):
    '''
    Function: converts words to tokens.
    Input: Word
    Output: tokens, attention-mask
    '''
    res = tokenizer.encode_plus(text, padding="max_length")
    return torch.tensor(res.input_ids), torch.tensor(res.attention_mask)


def read_dataset(file_name):
    '''
    Reading dataset and preprocessing it to get it in the desired forma
    '''
    res = []
    temp = pd.read_csv(file_name)
    for _, row in temp.iterrows():
        inp, target = row['transcription'], f'action {row["action"]} object {row["object"]} location {row["location"]}'
        res.append((tokenize(inp), tokenize(target)))
    return res


def evaluate(model, iterator, valid_ds_orig, pad_id):
    '''
    function: Evaluating the model
    Input: model, iterator, optimizer, pad_id
    Returns: epoch_loss, epoch_acc
    '''
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    final_pred = []
    # Predicted value
    comp_pred = []
    # Actual value
    total = len(valid_ds_orig)
    correct = 0
    with torch.no_grad():
        for (inp_ids, inp_mask), (target_ids, target_mask) in tqdm(iterator):
            model.to(device)
            inp_ids = inp_ids.to(device)
            inp_mask = inp_mask.to(device)
            target_ids[target_ids == pad_id] = -100  
            # needed to ignore padding from loss
            target_ids = target_ids.to(device)
            
            predictions = model(input_ids=inp_ids, attention_mask=inp_mask, labels=target_ids)
            loss = predictions.loss
            output = model.generate(input_ids = inp_ids)
            # Appending the batch to the final_pred after decoding
            for i in range(len(output)):
                final_pred.append(tokenizer.decode(output[i], skip_special_tokens=True))
            epoch_loss += loss.item()
    
    # Obtaining accuracy
    for i in range(len(valid_ds_orig)):
        comp_pred.append('action '+valid_ds_orig.iloc[i]['action']+' object '+valid_ds_orig.iloc[i]['object']+' location '+valid_ds_orig.iloc[i]['location'])
        correct += (comp_pred[i] == final_pred[i])
    print("Correct:",correct,"/",total)
    epoch_acc = (correct/total)*100.0
    return epoch_loss / len(iterator), epoch_acc


def run(model, tokenizer, root_dir):
    torch.cuda.empty_cache()
    seed_everything(SEED)
    
    # Maximum number of characters in a sentence. Set to 512.
    max_input_length = tokenizer.max_model_input_sizes[model_name]
    pad_token = tokenizer.pad_token
    
    # Padding helps prevent size mistmatch
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    
    # Reading dataset and preprocessing it to get it in the desired format
    valid_ds = read_dataset(f'{root_dir}/test.csv')
    
    # Dataloader
    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
    
    model = model.to(device)

    # Used to find accuracy
    valid_ds_orig = pd.read_csv(f'{root_dir}/test.csv')
    
    # Validating
    valid_loss, valid_acc = evaluate(model, valid_loader, valid_ds_orig, pad_id)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')

if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()
    # Path to the config file
    yml_path = f"{base_path}/config/config.yml"
    if not os.path.exists(yml_path):
        print("No such config file exists.")
        exit()
    with open(yml_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    # Extracting parameters from the config file.
    BATCH_SIZE = cfg["params"]["BATCH_SIZE"]
    model_name = cfg["params"]["model_name"]
    device = cfg["params"]["device"]

    SEED = 1234
    num_workers = 2
    # Path to the datset
    root_dir = f"{base_path}/task_data"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    # Imports the pretrained model and it's tokenizer
    print("Loading model and weights...")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    checkpoint = torch.load(f"{base_path}/weights/best_model.pth")
    model.load_state_dict(checkpoint)
    run(model, tokenizer, root_dir)
