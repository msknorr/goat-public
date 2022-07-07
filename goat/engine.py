import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('Agg')
import os
from datetime import datetime
import time


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FitterMaskRCNN:
    def __init__(self, model, device, config):
        """
        Engine for Fitting MaskRCNN model. For configs see config.
        :param model: MaskRCNN model
        :param device: torch.device, specified in config
        :param config: config file
        """
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.device = device
        self.model = model.to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        patience = 0
        losses_train = []
        losses_val = []
        lrs = []
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                lrs.append(lr)
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}\n')

            t = time.time()
            summary_losses = self.train_one_epoch(train_loader)
            losses_train.append(summary_losses[0].avg)
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_losses[0].avg:.5f} ' + \
                f'loss_classifier: {summary_losses[1].avg:.5f}, ' + \
                f'loss_box_reg: {summary_losses[2].avg:.5f}, ' + \
                f'loss_mask: {summary_losses[3].avg:.5f}, ' + \
                f'loss_objectness: {summary_losses[4].avg:.5f}, ' + \
                f'loss_rpn_box_reg: {summary_losses[5].avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_losses = self.validation(validation_loader)
            losses_val.append(summary_losses[0].avg)
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_losses[0].avg:.5f} ' + \
                f'loss_classifier: {summary_losses[1].avg:.5f}, ' + \
                f'loss_box_reg: {summary_losses[2].avg:.5f}, ' + \
                f'loss_mask: {summary_losses[3].avg:.5f}, ' + \
                f'loss_objectness: {summary_losses[4].avg:.5f}, ' + \
                f'loss_rpn_box_reg: {summary_losses[5].avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

            if summary_losses[0].avg < self.best_summary_loss:
                patience = 0
                print("saving best model")
                self.best_summary_loss = summary_losses[0].avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            else:
                patience += 1
                print("patience:", patience)
                if patience > 15:
                    print("//////////////// Patience. Training done.")
                    break

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_losses[0].avg)

            # plot and save train log
            fig, ax = plt.subplots(ncols=1)
            ax.plot(np.arange(len(losses_train)), np.array(losses_train), label="train loss")
            ax.plot(np.arange(len(losses_val)), np.array(losses_val), label="val loss")
            plt.legend()
            plt.grid()
            plt.ylim([0, 2])
            plt.savefig(f'{self.base_dir}/hist.png', dpi=144)
            plt.close(fig)
            np.save(f'{self.base_dir}/train_log.png', np.array([losses_train, losses_val, lrs]))
            self.epoch += 1

    def validation(self, val_loader):
        self.model.train()
        losses = [AverageMeter() for x in range(6)]
        t = time.time()
        for step, (images, targets, path) in enumerate(val_loader):

            if self.config.verbose:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {losses[0].avg:.5f}, ' + \
                 #  f'loss_classifier: {losses[1].avg:.5f}, ' + \
                 #  f'loss_box_reg: {losses[2].avg:.5f}, ' + \
                 #  f'loss_mask: {losses[3].avg:.5f}, ' + \
                 #  f'loss_objectness: {losses[4].avg:.5f}, ' + \
                 #  f'loss_rpn_box_reg: {losses[5].avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
            with torch.no_grad():
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                batch_size = len(images)
                output = self.model(images, targets)
                losses[0].update(sum(output.values()).detach().item(), batch_size)
                [losses[ii+1].update(output[k].item(), batch_size) for ii, k in enumerate(output.keys())]

        print("")
        return losses

    def train_one_epoch(self, train_loader):
        accumulation_steps = self.config.accumulation_steps

        self.model.train()
        losses = [AverageMeter() for x in range(6)]
        t = time.time()

        self.optimizer.zero_grad()
        total_loss = 0
        for step, (images, targets, path) in enumerate(train_loader):
            if self.config.verbose:
                if step % accumulation_steps == 0:
                    print(

                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {losses[0].avg:.5f}, ' + \
                     #  f'loss_classifier: {losses[1].avg:.5f}, ' + \
                     #  f'loss_box_reg: {losses[2].avg:.5f}, ' + \
                     #  f'loss_mask: {losses[3].avg:.5f}, ' + \
                     #  f'loss_objectness: {losses[4].avg:.5f}, ' + \
                     #  f'loss_rpn_box_reg: {losses[5].avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            batch_size = len(images)

            output = self.model(images, targets)
            sumloss = sum(output.values())
            (sumloss / accumulation_steps).backward()

            sumloss = sumloss.detach().cpu().numpy()
            total_loss += sumloss

            if (step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses[0].update(total_loss / accumulation_steps, batch_size)
                [losses[ii+1].update(output[k].item(), batch_size) for ii, k in enumerate(output.keys())]
                total_loss = 0

            if self.config.step_scheduler:
                self.scheduler.step()
        return losses

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        print("Checkpoint loaded for epoch:", self.epoch - 1)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')