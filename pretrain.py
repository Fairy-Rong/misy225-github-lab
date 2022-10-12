import logging
import os
import time
import torch, tqdm

from config import parse_args
from data_helper import create_dataloaders
from model import UniBERT
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, batch2cuda, MeanMetric, setup_ddp
from FGM import FGM, PGD
from torch.optim import swa_utils
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler

def pretrain(args):
    model = UniBERT(args).cuda()
    
    step = 0
    scaler = GradScaler()
    optimizer, scheduler, _ = build_optimizer(args, model)
    if os.path.exists(f'{args.savedmodel_path}/model.bin'):
        print(f'load from checkpoint {args.savedmodel_path}/model.bin')
        ckpt = torch.load(f'{args.savedmodel_path}/model.bin', map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt['step']
        optimizer.load_state_dict(ckpt['opt_state_dict'])
        scaler.load_state_dict(ckpt['scaler'])
        scheduler.load_state_dict(ckpt['scheduler'])

    train_dataloader, val_dataloader = create_dataloaders(args)
    

    nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    mlm_loss_metric = MeanMetric()
    mlm_acc_metric = MeanMetric()
    vtm_loss_metric = MeanMetric()
    vtm_acc_metric = MeanMetric()
    clip_loss_metric = MeanMetric()
    clip_acc_metric = MeanMetric()
    
    def _add_vtm_metrics(vtm_loss, vtm_acc):
        vtm_loss_metric.add(vtm_loss.item())
        vtm_acc_metric.add(vtm_acc.item())

    def _add_clip_metrics(clip_loss, clip_acc):
        clip_loss_metric.add(clip_loss.item())
        clip_acc_metric.add(clip_acc.item())

    def _add_mlm_metrics(mlm_loss, mlm_acc):
        mlm_loss_metric.add(mlm_loss.item())
        mlm_acc_metric.add(mlm_acc.item())

    while step < args.max_steps:
        train_dataloader.batch_sampler.init()
        bar = tqdm.tqdm(train_dataloader) if args.local_rank == 0 else train_dataloader
        for m in [mlm_loss_metric, mlm_acc_metric, vtm_loss_metric, vtm_acc_metric, clip_loss_metric, clip_acc_metric]:
            m.reset()
        for batch in bar:
            with autocast(enabled=True):
                batch = batch2cuda(batch)
                model.train()

                if step % 3 == 0:
                    # run vtm
                    mlm_loss, mlm_acc, vtm_loss, vtm_acc = model(batch, vtm_task=True, clip_task=False)
                    mlm_loss = mlm_loss.mean()

                    vtm_loss = vtm_loss.mean()
                    vtm_acc = vtm_acc.mean()

                    _add_vtm_metrics(vtm_loss, vtm_acc)

                    loss = mlm_loss + vtm_loss / 5.
                else:
                    # run clip
                    mlm_loss, mlm_acc, clip_loss, clip_acc = model(batch, vtm_task=False, clip_task=True)
                    mlm_loss = mlm_loss.mean()

                    clip_loss = clip_loss.mean()
                    clip_acc = clip_acc.mean()

                    _add_clip_metrics(clip_loss, clip_acc)

                    loss = mlm_loss + clip_loss / 2.

                mlm_acc = mlm_acc.mean()
                _add_mlm_metrics(mlm_loss, mlm_acc)
                
                # End the step
                scaler.scale(loss).backward()
                scheduler.step()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                
                del batch
            
            if args.local_rank == 0:
                bar.set_postfix(mlm_loss=mlm_loss_metric.val(), 
                                vtm_loss=vtm_loss_metric.val(), 
                                clip_loss=clip_loss_metric.val(),
                                mlm_acc=mlm_acc_metric.val(), 
                                vtm_acc=vtm_acc_metric.val(), 
                                clip_acc=clip_acc_metric.val(),
                                lr=optimizer.param_groups[0]['lr'])

            step += 1
            if step % args.print_steps == 0 and args.local_rank == 0:
                logging.info(f"Step {step}: mlm loss {mlm_loss_metric.val():.3f}, vtm loss {vtm_loss_metric.val():.3f}, clip_loss {clip_loss_metric.val():.3f}" + \
                            f"mlm acc {mlm_acc_metric.val():.3f}, vtm acc {vtm_acc_metric.val():.3f}, clip acc {clip_acc_metric.val():.3f}")

                # 5. save checkpoint
                torch.save({'step': step, 'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                            'opt_state_dict': optimizer.state_dict(),
                            'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(), },
                            f'{args.savedmodel_path}/model.bin')
                # torch.cuda.empty_cache()

    logging.info('Pretrain stopped')

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    if args.ddp:
        setup_ddp(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    pretrain(args)


if __name__ == '__main__':
    main()