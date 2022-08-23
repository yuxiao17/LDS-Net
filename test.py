import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *


class KDS_Player(BaseTrainer):
    def __init__(self, model, dataloader_test, plot_more=False):
        self.dataloader_test = dataloader_test
        self.model = model
        self.plot_more = plot_more
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        print(self.device)

    @no_grad
    def test(self, plot_dir='./images/samples-KinD'):
        self.model.eval()
        self.model.to(device=self.device)
        T=list()
        iter_start_time = time.time()
        for L_low_tensor, name in self.dataloader_test:
            start_time = time.time()
            L_low = L_low_tensor.to(self.device)
            R_low, I_low = self.model.decom_net(L_low)
            output = self.model.restore_net(L_low)
            end_time = time.time()
            cost = end_time - start_time
            T.append(cost)

            output_np = output.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            # Only plot result
            filepath = os.path.join(plot_dir, f'{name[0]}.png')
            split_point = [0, 3]
            img_dim = L_low_np.shape[1:]
            sample(output_np, split=split_point, figure_size=(1, 1),
                   img_dim=img_dim, path=filepath)

            if self.plot_more:
                R_low_np = R_low.detach().cpu().numpy()[0]
                I_low_np = I_low.detach().cpu().numpy()[0]
                split_point = [0, 3]
                img_dim = L_low_np.shape[1:]
                filepath = os.path.join(plot_dir, f'{name[0]}_R_LOW.png')
                sample(R_low_np, split=split_point, figure_size=(1, 1),
                       img_dim=img_dim, path=filepath)
                filepath2 = os.path.join(plot_dir, f'{name[0]}_I_LOW.png')
                sample(I_low_np, split=split_point, figure_size=(1, 1),
                       img_dim=img_dim, path=filepath2)
                # sample_imgs = np.concatenate((R_low_np, I_low_np, L_low_np,
                #                                output_np), axis=0)
                # filepath = os.path.join(plot_dir, f'{name[0]}_extra.png')
                # split_point = [0, 3, 4, 7, 10]
                # img_dim = L_low_np.shape[1:]

                # sample(sample_imgs, split=split_point, figure_size=(2, 2),
                #        img_dim=img_dim, path=filepath)
        Ts = sum(T)
        print('sum', Ts)
        print('mean',85/Ts)
        iter_end_time = time.time()
        timecost=iter_end_time - iter_start_time
        print('timecost',timecost)


class TestParser(BaseParser):
    def parse(self):
        self.parser.add_argument("-p", "--plot_more", default=False,
                                 help="Plot intermediate variables. such as R_images and I_images")
        self.parser.add_argument("-c", "--checkpoint", default="./checkpoints_nt/",
                                 help="Path of checkpoints")
        self.parser.add_argument("-i", "--input_dir", default="./images/inputc",
                                 help="Path of input pictures")
        self.parser.add_argument("-o", "--output_dir", default="./images/outputc/",
                                 help="Path of output pictures")
        # self.parser.add_argument("-b", "--b_target", default=0.75, help="Target brightness")
        # self.parser.add_argument("-u", "--use_gpu", default=True,
        #                         help="If you want to use GPU to accelerate")
        return self.parser.parse_args()

if __name__ == "__main__":
    model = KDS()
    parser = TestParser()
    args = parser.parse()

    input_dir = args.input_dir
    output_dir = args.output_dir
    plot_more = args.plot_more
    checkpoint = args.checkpoint
    decom_net_dir = os.path.join(checkpoint, "decom_net_l1v2.pth")
    restore_net_dir = os.path.join(checkpoint, "restore_net_5_last2.pth")


    pretrain_decom = torch.load(decom_net_dir)
    model.decom_net.load_state_dict(pretrain_decom)
    print(1)
    log('Model loaded from decom_net.pth')
    pretrain_resotre = torch.load(restore_net_dir)
    model.restore_net.load_state_dict(pretrain_resotre)
    print(2)
    log('Model loaded from restore_net.pth')

    log("Buliding Dataset...")
    dst = CustomDataset(input_dir)
    log(f"There are {len(dst)} images in the input direction...")
    dataloader = DataLoader(dst, batch_size=1)

    KinD = KDS_Player(model, dataloader, plot_more=plot_more)

    KinD.test(plot_dir=output_dir)