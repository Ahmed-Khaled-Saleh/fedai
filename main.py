import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import time

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

cs = ConfigStore.instance()
cs.store(group="data", name="cifar10", node= CIFAR10Config())
cs.store(group="data", name="cifar100", node= CIFAR100Config())
cs.store(group="data", name="mnist", node= MNISTConfig())
cs.store(group="data", name="mnist_rotated_batched", node= MNISTRotatedPatchedConfig())
cs.store(group="data", name="tinyimagenet", node= TinyImageNetConfig())

cs.store(group="model", name="lenet_fedavg", node= LeNetConfig())
cs.store(group="model", name="resnet_18", node= ResNetConfig())
cs.store(group="model", name="mobilenetv3_small", node= MobileNetConfig())
cs.store(group="model", name="efficientnet_b0", node= EfficientNetConfig())
cs.store(group="model", name="vit_small", node= ViTConfig())

cs.store(group="optimizer", name="sgd", node= BaseOptimizerConfig())
cs.store(group="optimizer", name="perfedavg_opt", node= PerFedavgOptimizerConfig())
cs.store(group="optimizer", name="pfedme_opt", node= pFedMeOptimizerConfig())
cs.store(group="optimizer", name="fedprox_opt", node= FedProxOptimizerConfig())
cs.store(group="optimizer", name="apfl_opt", node= APFLOptimizerConfig())

cs.store(group="algorithm", name="fedavg", node= FedAvgConfig())
cs.store(group="algorithm", name="sfmtl", node= SFMTLConfig())
cs.store(group="algorithm", name="fedprox", node= FedProxConfig())

cs.store(group="server", name="puhti", node= PuhtiConfig())

cs.store(name="base_config", node= MainConfig)

@hydra.main(version_base=None, config_name="base_config")
def main(cfg: MainConfig):
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