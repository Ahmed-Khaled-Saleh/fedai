
import torch 
import os 
from datetime import datetime 
from copy import deepcopy
import argparse
from torch import nn

from fedai.data import init_data, IMG_DATA_CONFIGS
from fedai.client_selector import BaseClientSelector
from fedai.wandb_writer import WandbWriter
from fedai.utils import init_server, get_criterion

from omegaconf import OmegaConf 
from huggingface_hub import login 
from dotenv import load_dotenv 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
    parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)
    

    parser.add_argument('--lr', type=float, help='Learning rate local', required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False)
    parser.add_argument('--optimizer', type=str, help='Optimizer', required=False)
    parser.add_argument('--client_cls', type=str, help='Client class', required=False)
    parser.add_argument('--agg', type=str, help='Aggregation', required=False)
    parser.add_argument('--lambda_', type=str, help='lambda for fedu and dmtl', required=False)
    parser.add_argument('--alpha', type=str, help='alpha for dmtl', required=False)
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)
        key = os.getenv("WANDB_API_KEY", None)
        hf_secret = os.getenv("HF_SECRET_CODE", None)

        if key:
            os.environ["WANDB_API_KEY"] = key
        if hf_secret:
            os.environ["HF_SECRET_CODE"] = hf_secret     

    try:
        with open(args.config, 'r') as file:
            cfg = OmegaConf.load(file)
    except:
        print("Invalid config file path")


    cfg.now = args.timestamp 

    cfg.optimizer.lr = float(args.lr) if args.lr else cfg.optimizer.lr
    cfg.data.batch_size = int(args.batch_size) if args.batch_size else cfg.data.batch_size
    cfg.optimizer.name = args.optimizer if args.optimizer else cfg.optimizer.name

    cfg.client_cls = args.client_cls if args.client_cls else cfg.client_cls

    cfg.agg = args.agg if args.agg else cfg.agg

    if  cfg.client_cls in ["pFedMe", "Fedu", "DMTL"]:
        cfg.lambda_ = float(args.lambda_) if args.lambda_ else cfg.lambda_

    if cfg.client_cls == "DMTL":
        cfg.alpha = float(args.alpha) if args.alpha else cfg.alpha

    
    cfg.data.partitioner.kwargs.update({"partition_by": IMG_DATA_CONFIGS[cfg.data.name].y})
    fds = init_data(cfg)

    client_selector = BaseClientSelector(cfg)
    loss_fn = get_criterion(None)
    writer = WandbWriter(cfg)

    server = init_server(algo_name=cfg.client_cls, 
                          config=cfg, 
                          selector=client_selector, 
                          criterion=loss_fn, 
                          fds=fds, 
                          writer=writer)
    
    server.train()
