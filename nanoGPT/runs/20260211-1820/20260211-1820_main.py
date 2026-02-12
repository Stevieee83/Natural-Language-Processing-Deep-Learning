# Imports the PyTorch libraries to the script
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
from datetime import datetime
import shutil

import logging
import hydra
from omegaconf import DictConfig

from model import BigramLanguageModel

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info('Config File')
    log.info(cfg)

    os.makedirs(cfg.directories.log_dir, exist_ok=True)

    # Creates the filepath directories to save the experiment results to
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    overall_path = os.path.join(cfg.directories.log_dir, run_id)
    os.makedirs(overall_path, exist_ok=True)

    # Copies the Python script file for the experiments into the file path directory created
    shutil.copyfile(
        __file__, os.path.join(overall_path, run_id + "_" + os.path.basename(__file__))
    )

    # Copies the Hydra .yaml config file for the experiments into the file path directory created
    shutil.copyfile(
        './config/config.yaml', os.path.join(overall_path, run_id + "_config.yaml")
    )

    # Sets the device and the random seeds
    if torch.cuda.is_available():
        # Sets the deice to CUDA GPU
        device = 'cuda'
        # Set random seed for GPU
        torch.cuda.manual_seed(cfg.training.seed)
        torch.cuda.manual_seed_all(cfg.training.seed)

    else:
        # Sets the deice to CUDA GPU
        device = 'cpu'
        # Set random seed for CPU
        torch.cuda.manual_seed(cfg.training.seed)

    # Logs the device used to the screen
    log.info(f"Device: {device}")

    # Read it in to inspeact it
    with open(cfg.directories.text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique charecters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from charectors to integers
    stoi = { ch:i for i, ch in enumerate(chars)}
    itos = { i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]           # Encoder: take a string, output list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: take a list of integers, output a string

    # Initilises the input data a PyTorch torch tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # Let's now split up the data into train and validation sets
    n = int(0.9*len(data))      # First 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # Cret=ates the model and sends it to the GPU
    model = BigramLanguageModel(vocab_size=vocab_size,
                                device=device,
                                cfg=cfg)       # removed vocab_size from the contructor as it is a global variable at the top
    m = model.to(device)

    # Create a PyTorch optimisation object
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for iter in range(cfg.training.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % cfg.training.eval_interval == 0:
            losses = estimate_loss(model, cfg.training.eval_iters, device, cfg.training.block_size, cfg.training.batch_size, train_data, val_data)
            log.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', device, cfg.training.block_size, cfg.training.batch_size, train_data)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    log.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Makes predictions with the trained model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    result_txt = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    log.info(result_txt)

    model_path_full = os.path.join(overall_path, cfg.directories.model_path)

    torch.save(model.state_dict(), model_path_full)

# Python method to load a batch to the script
def get_batch(split, device, block_size, batch_size, train_data, val_data=None):

    """Python method to load a batch of data to the language model"""

    # generate a small batch of data of inputs x and y targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Python method with the PyTorch no_grad decorator to estimate the loss of the model
@torch.no_grad()
def estimate_loss(model, eval_iters, device, block_size, batch_size, train_data, val_data):

    """Python method to estimate the loss of the language model"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, device, block_size, batch_size, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.mean()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ ==  "__main__":
    main()
