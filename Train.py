import random

import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip

from Modules.MriDataset import MriDataset
from Modules.Utils import init
from Network import Network

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self) -> None:
        self.args, self.checkpoints_dir, self.tensorboard = init()

        self.net = Network().to("cuda" if self.args.gpu else "cpu")

        self.optimizer = RMSprop([{'params': self.net.parameters(), 'initial_lr': self.args.lr}], lr=self.args.lr)

        self.train_dataset = MriDataset(root=self.args.root, split=self.args.train_dataset,
                                        transform=Compose([RandomHorizontalFlip(p=0.4), RandomVerticalFlip(p=0.4)]))
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)

        self.eval_dataset = MriDataset(root=self.args.root, split=self.args.eval_dataset)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1, shuffle=False, pin_memory=True)

        self.train_steps = self.eval_steps = 0

        self.scales = 4

    def train(self) -> None:
        start_epoch, best_bpsp = self.load_checkpoints()

        scheduler = MultiStepLR(optimizer=self.optimizer, milestones=self.args.lr_decay_milestone,
                                gamma=self.args.lr_decay_factor, last_epoch=start_epoch - 1)

        for epoch in range(start_epoch, self.args.max_epoch):
            self.train_one_epoch()
            scheduler.step()

            if epoch % self.args.eval_epochs == 0:
                bpp = self.eval_one_epoch()
                if bpp < best_bpsp:
                    self.save_ckpt(epoch=epoch, best_bpp=bpp)
                    best_bpsp = bpp

    @torch.no_grad()
    def eval_one_epoch(self) -> float:
        self.net.eval()
        self.eval_steps += 1

        average_bpp_list = [0., ] * self.scales

        for slices in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), ncols=50):

            slices = slices.to("cuda" if self.args.gpu else "cpu")

            slices = [slices[:, i, :, :] for i in range(slices.shape[1])]

            H_t_minus1_3 = H_t_minus1_2 = H_t_minus1_1 = H_t_minus1_0 = None

            bpp_list = [torch.zeros(1), ] * 3

            for x in slices:
                x = x.unsqueeze(dim=1)

                outputs = self.net(X_t=x, H_t_minus1_3=H_t_minus1_3, H_t_minus1_2=H_t_minus1_2,
                                   H_t_minus1_1=H_t_minus1_1, H_t_minus1_0=H_t_minus1_0)

                H_t_minus1_0, H_t_minus1_1, H_t_minus1_2, H_t_minus1_3 = outputs["LatentFeats"]

                for i, bpp in enumerate(outputs["Bpp"]):
                    bpp_list[i] += bpp

            for i in range(self.scales):
                average_bpp_list[i] += bpp_list[i]

        average_bpp = sum(average_bpp_list) / len(self.eval_dataloader)

        self.tensorboard.add_scalars(main_tag="Eval/Bpp", global_step=self.eval_steps,
                                     tag_scalar_dict={"Scale_{}".format(i): average_bpp_list[i].cpu() / len(self.eval_dataloader)for i in range(self.scales)})

        self.tensorboard.add_scalar(tag="Eval/Total Bpp", global_step=self.eval_steps, scalar_value=average_bpp.cpu())

        return average_bpp

    def train_one_epoch(self):
        self.net.train()
        for slices in tqdm(self.train_dataloader, total=len(self.train_dataloader), ncols=50):
            self.train_steps += 1

            _, num_slices, _, _ = slices.shape
            assert num_slices >= self.args.training_slices

            start_slice = random.randint(0, num_slices - self.args.training_slices)
            slices = slices[:, start_slice:, :, :].to("cuda" if self.args.gpu else "cpu")

            slices = [slices[:, i, :, :] for i in range(self.args.training_slices)]

            H_t_minus1_3 = H_t_minus1_2 = H_t_minus1_1 = H_t_minus1_0 = None

            bpp_list = [torch.zeros(1).to("cuda" if self.args.gpu else "cpu"), ] * self.scales

            for x in slices:
                x = x.unsqueeze(dim=1)

                outputs = self.net(X_t=x, H_t_minus1_3=H_t_minus1_3, H_t_minus1_2=H_t_minus1_2,
                                   H_t_minus1_1=H_t_minus1_1, H_t_minus1_0=H_t_minus1_0)

                H_t_minus1_0, H_t_minus1_1, H_t_minus1_2, H_t_minus1_3 = outputs["LatentFeats"]

                for i, bpp in enumerate(outputs["Bpp"]):
                    bpp_list[i] += bpp

            loss = sum(bpp_list)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.train_steps % 10 == 0:
                self.tensorboard.add_scalars(main_tag="Train/Bpp", global_step=self.train_steps,
                                             tag_scalar_dict={"Scale_{}".format(i): bpp_list[i].cpu() for i in range(self.scales)})

    def save_ckpt(self, epoch: int, best_bpp: float) -> None:
        checkpoint = {
            "network": self.net.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "best_bpp": best_bpp
        }

        torch.save(checkpoint, '%s/model_%.3d.pth' % (self.checkpoints_dir, epoch))
        print("\n======================Saving model {0}======================".format(str(epoch)))

    def load_checkpoints(self) -> tuple:
        best_bpp = 1e9
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")
            self.net.load_state_dict(ckpt["network"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except:
                print("Can not find some optimizers params, just ignore")
            try:
                best_bpp = ckpt['best_bpp']
            except:
                print("Can not find the record of the best bpp, just set it to 1e9 as default")
            try:
                start_epoch = ckpt["epoch"] + 1
            except:
                start_epoch = 0
        elif self.args.pretrained:
            ckpt = torch.load(self.args.pretrained)
            print("\n===========Load network weights {0}===========\n".format(self.args.pretrained))
            # load codec weights
            pretrained_dict = ckpt["network"]
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
            start_epoch = 0
        else:
            print("\n===========Training from scratch===========\n")
            start_epoch = 0
        return start_epoch, best_bpp


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
