import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import timm
from transformers import AutoModel
from safetensors import safe_open
from core.model import create_resnet_gn

# import detectors
# Load model directly
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# Use a pipeline as a high-level helper

from core.eval import evaluate_ori, evaluate_ood, evaluate_train_head
from core.calibration import calibration_ori
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn
from core.setada import *

import time

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # configure base model
    if 'BN' in cfg.MODEL.ARCH:
        if cfg.CORRUPTION.DATASET == 'cifar10' and cfg.MODEL.ARCH == 'WRN2810_BN':
            # use robustbench
            model = 'Standard'
            base_model = load_model(model, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).to(device)
        elif cfg.CORRUPTION.DATASET == 'cifar100' or cfg.CORRUPTION.DATASET == 'tin200':
            base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            # print(base_model)
            if cfg.CORRUPTION.DATASET == 'cifar100':
                print('Building model for cifar100...')
                ckpt = torch.load("/root/workspace/tea/tea/ckpt/cifar100/cifar100_checkpoint.pth")
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['state_dict'].items()}
            else:
                print('Building model for tin200...')
                base_model.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False).to(device)
                state_dict = torch.load("/root/workspace/tea/z_wideresnet_tin200_robustbench_best.pth")
            base_model.load_state_dict(state_dict)
            print('Model loaded...')
        elif cfg.CORRUPTION.DATASET == 'pacs' or cfg.CORRUPTION.DATASET == 'mnist' :
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError
    elif 'GN' in cfg.MODEL.ARCH:
        base_model = create_resnet_gn(cfg.CORRUPTION.NUM_CLASSES).to(device)
        if cfg.CORRUPTION.DATASET == 'cifar100':
            ckpt = torch.load("/root/workspace/tea/resnet50_gn_cifar100_robustbench.pth")
            print('Model loaded...')
        elif cfg.CORRUPTION.DATASET == 'cifar10':
            ckpt = torch.load("/root/workspace/tea/resnet50_gn_cifar10_robustbench.pth")
            print('Model loaded...')
        elif cfg.CORRUPTION.DATASET == 'tin200':
            ckpt = torch.load("/root/workspace/tea/resnet50_gn_tin200_robustbench_best.pth")
            print('Model loaded...')
        base_model.load_state_dict(ckpt)
    else:
        raise NotImplementedError

    # configure tta model
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eta":
        logger.info("test-time adaptation: ETA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eata":
        logger.info("test-time adaptation: EATA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "sar":
        logger.info("test-time adaptation: SAR")
        model = setup_sar(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "shot":
        logger.info("test-time adaptation: SHOT")
        model = setup_shot(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "pl":
        logger.info("test-time adaptation: PL")
        model = setup_pl(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "schrodinger":
        logger.info("test-time adaptation: SCHRODINGER")
        model = setup_schrodinger(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    # train schrodinger head
    evaluate_train_head(model, cfg, logger, device)


    # evaluate on each severity and type of corruption in turn
    # evaluate_ood(model, cfg, logger, device)
    

if __name__ == '__main__':
    main()
