import numpy as np
import pandas as pd
import math

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from torchvision.utils import save_image
from alive_progress import alive_bar
import time
import os


from core.data import load_data, load_dataloader

def clean_accuracy(model, x, y, batch_size = 100, device = None, ada=None, if_adapt=True, if_vis=False, logger=None, info=None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        energes_list=[]
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            if ada == 'source':
                output = model(x_curr)

            else:
                output = model(x_curr, if_adapt=if_adapt)

            acc += (output.max(1)[1] == y_curr).float().sum()
            print(f'batch acc: {acc.item() / (counter+1) / batch_size}, counter: {100*counter/n_batches}% ' + info)

    
    return acc.item() / x.shape[0]

def clean_accuracy_loader(model, test_loader, device=None, ada=None, if_adapt=True, if_vis=False, logger=None):
    test_loss = 0
    correct = 0
    index = 1
    total_step = math.ceil(len(test_loader.dataset) / test_loader.batch_size)
    with torch.no_grad():
        for counter, (data, target) in enumerate(test_loader):
            logger.info("Test Batch Process: {}/{}".format(index, total_step))
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                if ada == 'source':
                    output = model(data)
                else:
                    output = model(data, if_adapt=if_adapt)
            test_loss += F.cross_entropy(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            index = index + 1
            
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc

def evaluate_ood(model, cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        res = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        for c in range(len(cfg.CORRUPTION.TYPE)):
            # if cfg.CORRUPTION.TYPE[c] != 'shot_noise':
            #     print(f"skip {cfg.CORRUPTION.TYPE[c]}")
            #     continue
            for s in range(len(cfg.CORRUPTION.SEVERITY)):
                # if cfg.CORRUPTION.SEVERITY[s] != 5:
                #     continue
                # try:
                model.reset()
                logger.info("resetting model") # load load_state_dict and optimizer
                # except:
                #     logger.warning("not resetting model")

                if cfg.MODEL.ADAPTATION == 'schrodinger':
                    # load head state_dict and optimizer
                    model.reset_schrodinger_head(head_path=cfg.SCHRODINGER.LOAD_HEAD_DIR, ada_head_during_test=cfg.SCHRODINGER.ADA_HEAD_DURING_TEST)

                x_test, y_test = load_data(cfg.CORRUPTION.DATASET+'c', cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, False,
                                            [cfg.CORRUPTION.TYPE[c]])
                x_test, y_test = x_test.to(device), y_test.to(device)
                acc = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE,  ada=cfg.MODEL.ADAPTATION, if_adapt=True, logger=logger, info=f"[{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]")           
                logger.info(f"acc % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {acc:.2%}")
                res[s, c] = acc

            logger.info(f"mean acc % [{cfg.CORRUPTION.TYPE[c]}]: {np.mean(res[:, c]):.2%}")

        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))
    
    elif cfg.CORRUPTION.DATASET == 'mnist':
        _, _, _, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
        acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION,  if_adapt=True, if_vis=True)
        logger.info("Test set Accuracy: {}".format(acc))
    
    elif cfg.CORRUPTION.DATASET == 'pacs':
        accs=[]
        for target_domain in cfg.CORRUPTION.TYPE:
            x_test, y_test = load_data(data = cfg.CORRUPTION.DATASET, data_dir=cfg.DATA_DIR, shuffle = True, corruptions=target_domain)
            x_test, y_test = x_test.to(device), y_test.to(device)
            out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=True, logger=logger, info=f"[{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]")
            if cfg.MODEL.ADAPTATION == 'energy':
                acc = out[0]
            else:
                acc = out
            logger.info(f"acc % [{target_domain}]: {acc:.2%}")
            accs.append(acc)
        acc_mean=np.array(accs).mean()
        logger.info(f"mean acc:%: {acc_mean:.2%}")
    else:
        raise NotImplementedError

def evaluate_adv(base_model, model, cfg, logger, device):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
        x_test, y_test = x_test.to(device), y_test.to(device)
        adversary = AutoAttack(base_model, norm='L2', eps=0.5, version='custom', attacks_to_run=['apgd-ce'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test)
        acc = clean_accuracy(model, x_adv, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, if_adapt=True)
        logger.info("acc:".format(acc))

def evaluate_ori(model, cfg, logger, device):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        if 'cifar' in cfg.CORRUPTION.DATASET:
            x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
            x_test, y_test = x_test.to(device), y_test.to(device)
            out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
            if cfg.MODEL.ADAPTATION == 'energy':
                acc, energes = out
            else:
                acc = out
            logger.info("Test set Accuracy: {}".format(acc))

        elif 'pacs' in cfg.CORRUPTION.DATASET:
            pass
        else:  #tin200
            _,_,_,test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
            acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
            logger.info("Test set Accuracy: {}".format(acc))







def train_schrodingerHead(model, x, batch_size = 100, logger=None, device = None, train_epochs=100, save_path=None):
    if device is None:
        device = x.device
    n_batches = math.ceil(x.shape[0] / batch_size)

    for counter in range(n_batches):
        x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
        a_item = model.get_a(x_curr, counter)
    logger.info("a: {}".format(a_item))

    # a_ = -1000
    for epoch in range(train_epochs):
        perm = torch.randperm(x.size(0), device=x.device)
        x = x[perm]
        with alive_bar(n_batches, title="epoch: "+ str(epoch) + " a: " + str(a_item)) as bar:
            for counter in range(n_batches):
                time.sleep(0.1)
                bar()
                x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
                schrodinger_head, _, optimizer_head = model.train_schrodinger_head(x_curr)
                

        if (epoch+1) % 5 == 0 or (epoch+1) <= 20:
            model.visualize_schrodinger(device=device, epoch=epoch, save_path=save_path.replace('schrodinger_head', 'visualize_schrodinger'))

        if (epoch+1) % 50 == 0:
            save_path_ = save_path + 'schrodingerHead_' + str(epoch+1) + '.pth'
            os.makedirs(os.path.dirname(save_path_), exist_ok=True)
            torch.save({'model_state_dict': schrodinger_head.state_dict(),
                        'a': a_item,
                        'optimizer_state_dict': optimizer_head.state_dict(),}, 
                        save_path_)
    

def train_schrodingerHead_loader(model, logger=None, device=None, train_epochs=100, save_path=None, root=None, dataset=None, batch_size=100):
    _,_,_,train_loader = load_dataloader(root=root, dataset=dataset, batch_size=batch_size, if_shuffle=False, logger=logger)
    for counter, (data, _) in enumerate(train_loader):
        data = data.to(device)
        a_item = model.get_a(data, counter)
    logger.info("a: {}".format(a_item))

    # a_ = a_item
    for epoch in range(train_epochs):
        _,_,_,train_loader = load_dataloader(root=root, dataset=dataset, batch_size=batch_size, if_shuffle=True, logger=logger)
        total_step = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        with alive_bar(total_step, title="epoch: "+ str(epoch) + " a: " + str(a_item) ) as bar:
            for counter, (data, _) in enumerate(train_loader):
                time.sleep(0.1)
                bar()
                data = data.to(device)
                schrodinger_head, a, optimizer_head = model.train_schrodinger_head(data)

        if (epoch+1) % 5 == 0 or (epoch+1) <= 20:
            model.visualize_schrodinger(device=device, epoch=epoch, save_path=save_path.replace('schrodinger_head', 'visualize_schrodinger'))

        if (epoch+1) % 50 == 0:
            save_path_ = save_path + 'schrodingerHead_' + str(epoch+1) + '.pth'
            os.makedirs(os.path.dirname(save_path_), exist_ok=True)
            torch.save({'model_state_dict': schrodinger_head.state_dict(),
                        'a': a,
                        'optimizer_state_dict': optimizer_head.state_dict(),}, 
                        save_path_)

def evaluate_train_head(model, cfg, logger, device):
        model.set_head_optimizer() # important
        if 'cifar' in cfg.CORRUPTION.DATASET:
            x_train, y_train = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
            x_train, y_train = x_train.to(device), y_train.to(device)
            train_schrodingerHead(model, x_train, cfg.OPTIM.BATCH_SIZE, logger=logger, train_epochs=cfg.SCHRODINGER.TRAIN_SCHRODINGERHEAd_EPOCHS, save_path=cfg.SCHRODINGER.SAVE_HEAD_DIR+'cifar/')
            logger.info("Train schrodinger head finish.")
        elif 'pacs' in cfg.CORRUPTION.DATASET:
            pass
        else:
            train_schrodingerHead_loader(model, logger=logger, device=device, root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, train_epochs=cfg.SCHRODINGER.TRAIN_SCHRODINGERHEAd_EPOCHS, save_path=cfg.SCHRODINGER.SAVE_HEAD_DIR+'other/')
            logger.info("Train schrodinger head finish.")