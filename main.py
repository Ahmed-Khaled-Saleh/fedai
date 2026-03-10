import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import time
import os
from fedai.cfgs import MainConfig
from fedai.cfgs.data import *
from fedai.cfgs.models import *
from fedai.cfgs.optimizers import *
from fedai.cfgs.algos import *
from fedai.cfgs.gpu_servers import *

from fedai.data import init_data
from fedai.models import create_model
from fedai.client_selector import BaseClientSelector
from fedai.wandb_writer import WandbWriter
from fedai.utils import init_server, get_criterion
from dotenv import load_dotenv
cs = ConfigStore.instance()

cs.store(group="data", name="mnist", node= MNISTConfig())
cs.store(group="data", name="mnist_rotated_batched", node= MNISTRotatedPatchedConfig())
cs.store(group="data", name="fashionmnist", node= FashionMNISTConfig())
cs.store(group="data", name="cifar10", node= CIFAR10Config())
cs.store(group="data", name="cifar100", node= CIFAR100Config())
cs.store(group="data", name="tinyimagenet", node= TinyImageNetConfig())

cs.store(group="model", name="lenet", node= LeNetConfig())
cs.store(group="model", name="resnet", node= ResNetConfig())
cs.store(group="model", name="mobilenetv3", node= MobileNetConfig())
cs.store(group="model", name="efficientnet", node= EfficientNetConfig())
cs.store(group="model", name="vit", node= ViTConfig())

cs.store(group="optimizer", name="sgd", node= SGDConfig())
cs.store(group="optimizer", name="adam", node= AdamConfig())
cs.store(group="optimizer", name="adamw", node= AdamWConfig())
cs.store(group="optimizer", name="perfedavg", node= PerFedavgOptimizerConfig())
cs.store(group="optimizer", name="pfedme", node= pFedMeOptimizerConfig())
cs.store(group="optimizer", name="fedprox", node= FedProxOptimizerConfig())
cs.store(group="optimizer", name="apfl", node= APFLOptimizerConfig())

cs.store(group="algorithm", name="fedavg", node= FedAvgConfig())
cs.store(group="algorithm", name="fedavg_ft", node= FedAvgFTConfig())
cs.store(group="algorithm", name="pfedme", node= pFedMeConfig())
cs.store(group="algorithm", name="fedu", node= FedUConfig())
cs.store(group="algorithm", name="sfmtl", node= SFMTLConfig())
cs.store(group="algorithm", name="perfedavg", node= PerFedAvgConfig())
cs.store(group="algorithm", name="ditto", node= DittoConfig())
cs.store(group="algorithm", name="fedprox", node= FedProxConfig())
cs.store(group="algorithm", name="apfl", node= APFLConfig())
cs.store(group="algorithm", name="fedala", node= FedALA())
cs.store(group="algorithm", name="ifca", node= IFCAConfig())
cs.store(group="algorithm", name="fedper", node= FedPerConfig())
cs.store(group="algorithm", name="lgfedavg", node= LGFedAvgConfig())
cs.store(group="algorithm", name="fedrep", node= FedRepConfig())
cs.store(group="algorithm", name="fedrod", node= FedRodConfig())
cs.store(group="algorithm", name="fedbabu", node= FedBabuConfig())
cs.store(group="algorithm", name="gpfl", node= FedGPFLConfig())
cs.store(group="algorithm", name="feddbe", node= FedDBEConfig())

cs.store(group="server", name="puhti", node= PuhtiConfig())

cs.store(name="base_config", node= MainConfig)

@hydra.main(version_base=None, config_name="base_config")
def main(cfg: MainConfig):
    load_dotenv()

    print(f"Algorithm: {cfg.algorithm.name}")
    print(f"Model: {cfg.model.name}")
    print(f"Data: {cfg.data.name}")
    print(f"Server: {cfg.server.name}")

    fds = init_data(cfg.data)
    cfg.now = time.strftime("%Y-%m-%d_%H-%M-%S")

    client_selector = BaseClientSelector(cfg)
    criterion = get_criterion(None)
    writer = WandbWriter(cfg)

    server = init_server(algo_name= cfg.algorithm.name,
                         config= cfg, 
                         selector= client_selector,
                         criterion= criterion,
                         fds= fds,
                         writer= writer)
    server.train()

if __name__ == "__main__":
    main()