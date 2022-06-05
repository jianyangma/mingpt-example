from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    
    attention_model = GPT(mconf)
    
    return attention_model

def initialize_synthesizer_model(mconf):
    attention_model = None

    mconf.synthesizer = True
    attention_model = GPT(mconf)
    
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model):
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ###     3. An output path writing_params_path for the model parameters
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu')) to load pretrained model 

    trainer_obj = None #Trainer object
    tconf = None #TrainerConfig object

    text = open(finetune_corpus_path, encoding='utf-8').read()

    finetune_dataset = NameDataset(text, pretrain_dataset)
    if reading_params_path is None:
        tconf = TrainerConfig(max_epochs=75, batch_size=256, learning_rate=6e-4,
                              lr_decay=True, warmup_tokens=512 * 20, final_tokens=200 * len(pretrain_dataset) * block_size,
                              num_workers=4)
        trainer_obj = Trainer(model, finetune_dataset, None, tconf)

    else:
        pretrained_model = torch.load(reading_params_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_model)

        tconf = TrainerConfig(max_epochs=10, batch_size=256, learning_rate=6e-4,
                              lr_decay=True, warmup_tokens=512 * 20, final_tokens=200 * len(pretrain_dataset) * block_size,
                              num_workers=4)

        trainer_obj = Trainer(model, finetune_dataset, None, tconf)
    
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model):
    ### - Given:
    ###     1. A corpus specified in pretrain_corpus_path
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=4

    trainer_obj = None #Trainer object
    tconf = None #TrainerConfig object

    tconf = TrainerConfig(max_epochs=650, batch_size=128, learning_rate=6e-3,
                          lr_decay=True, warmup_tokens=512 * 20, final_tokens=200 * len(pretrain_dataset) * block_size,
                          num_workers=4)

    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    trainer_obj.train()
    torch.save(model.state_dict(), writing_params_path)
    return
