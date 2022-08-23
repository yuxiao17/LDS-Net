import  os
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_ssim
from dataloader import *
from torchvision.models import vgg16_bn
from torchvision.models import vgg16,vgg19
import torchvision.transforms as transforms

# only conv5_4
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg_model=vgg19(pretrained=True).features[:].to(device)
        vgg_model.eval()
        for param in vgg_model.parameters():
            param.requires_grad=False

        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '35': "relu5_4"

        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        # self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.weight = [0.1,0.1,1.0,1.0,1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            # print("vgg_layers name:",name,module)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        # print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        #guiyi
        transforms1 = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        output = transforms1(output)
        gt = transforms1(gt)

        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)
        s= sum(loss)
        # return sum(loss), output_features  # /len(loss)
        return s
vg19=LossNetwork()

####5 block
class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[0,1,2,3,4]
        weights=[1,1,1,1,1]
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # inputs = F.normalize(inputs, mean, std)
        # targets = F.normalize(targets, mean, std)
        transforms1=transforms.Compose([
            transforms.Normalize(mean = (0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225))
        ])
        inputs=transforms1(inputs)
        targets=transforms1(targets)
        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += F.mse_loss(lhs, rhs) * w

        return loss


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()

vg16=FeatureLoss()
# def perceptual_loss(x, y):
#     F.mse_loss(x, y)
#
#
# def PerceptualLoss(blocks, weights, device):
#     return FeatureLoss(perceptual_loss, blocks, weights, device)


Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def feature_map_hook(*args, path=None):
    feature_maps = []
    for feature in args:
        feature_maps.append(feature)
    feature_all = torch.cat(feature_maps, dim=1)
    fmap = feature_all.detach().cpu().numpy()[0]
    fmap = np.array(fmap)
    fshape = fmap.shape
    num = fshape[0]
    shape = fshape[1:]
    sample(fmap, figure_size=(2, num//2), img_dim=shape, path=path)
    return fmap

# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()
        # self.ssim_loss = pytorch_ssim.SSIM()
        # loss_ssim = 1 - self.ssim_loss(R_low, R_high)

    def reflectance_similarity(self, R_low, R_high):
        return torch.mean(torch.abs(R_low - R_high))
    
    def illumination_smoothness(self, I, L, name='low', hook=-1):
        # L_transpose = L.permute(0, 2, 3, 1)
        # L_gray_transpose = 0.299*L[:,:,:,0] + 0.587*L[:,:,:,1] + 0.114*L[:,:,:,2]
        # L_gray = L.permute(0, 3, 1, 2)
        L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
        L_gray = L_gray.unsqueeze(dim=1)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
        I_gradient_y = gradient(I, "y")
        L_gradient_y = gradient(L_gray, "y")
        Denominator_y = torch.max(L_gradient_y, epsilon)
        y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
        mut_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I, L_gray, epsilon, I_gradient_x+I_gradient_y, Denominator_x+Denominator_y, 
                            x_loss+y_loss, path=f'./waterimage/samples-features/ilux_smooth_{name}_epoch{hook}.png')
        return mut_loss
    
    def mutual_consistency(self, I_low, I_high, hook=-1):
        low_gradient_x = gradient(I_low, "x")
        high_gradient_x = gradient(I_high, "x")
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = gradient(I_low, "y")
        high_gradient_y = gradient(I_high, "y")
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        if hook > -1:
            feature_map_hook(I_low, I_high, low_gradient_x+low_gradient_y, high_gradient_x+high_gradient_y, 
                    M_gradient_x + M_gradient_y, x_loss+ y_loss, path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss

    def reconstruction_error(self, R_low, R_high, I_low_3, I_high_3, L_low, L_high):
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))
        # recon_loss_l2h = torch.mean(torch.abs(R_high * I_low_3 -  L_low))
        # recon_loss_h2l = torch.mean(torch.abs(R_low * I_high_3 - L_high))
        return recon_loss_high + recon_loss_low # + recon_loss_l2h + recon_loss_h2l

    def icloss(self,I_low,I_high,L_low,L_high):
        ilow=torch.mean(I_low)
        ihigh=torch.mean(I_high)
        llow=torch.mean(L_low)
        lhigh=torch.mean(L_high)
        x=torch.div(ilow,ihigh)
        y=torch.div(llow,lhigh)
        loss=torch.abs(x-y)
        return loss

    def grad_loss(self, low, high, hook=-1):
        x_loss = F.mse_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.mse_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high, hook=-1):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        #network output
        recon_loss = self.reconstruction_error(R_low, R_high, I_low_3, I_high_3, L_low, L_high)
        # loss_ssim = 1 - self.ssim_loss(R_high, L_high)+0.6(1-self.ssim_loss(R_low, R_high))
        # equal_R_loss = self.reflectance_similarity(R_low, R_high)
        # equal_R_loss = self.reflectance_similarity(R_low, R_high)+0.5*self.reflectance_similarity(L_high, R_high)
        # grad_lh=self.grad_loss(R_low,R_high)
        # i_mutual_loss = self.mutual_consistency(I_low, I_high, hook=hook)
        # ilux_smooth_loss = self.illumination_smoothness(I_low, L_low, hook=hook) + \
        #              self.illumination_smoothness(I_high, L_high, name='high', hook=hook)
        lhloss=(self.icloss(I_low,I_high,L_low,L_high))*0.1
        # equa=0.009 * equal_R_loss
        # iloss=0.05* ilux_smooth_loss
        # vg19loss = (vg19(R_low, R_high))*0.001
        # l2=F.mse_loss(R_low,R_high)
        # ss=(1-self.ssim_loss(R_low,R_high))*0.01
        # decom_loss = recon_loss + 0.009 * equal_R_loss + 0.2 * i_mutual_loss + 0.15 * ilux_smooth_loss+0.1 * ilux_smooth_loss
        # decom_loss = recon_loss + equa+lhloss+iloss
        decom_loss =recon_loss+lhloss
        # print('recon',recon_loss)
        # print('l2',l2)
        # print('ss',ss)
        # print('equal',equa)
        # # print('vg19',vg19loss)
        # print('lh',lhloss)
        # print('smooth',iloss)

        return decom_loss


class Illum_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_loss(self, low, high, hook=-1):
        x_loss = F.l1_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.l1_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def forward(self, I_low, I_high, hook=-1):
        loss_grad = self.grad_loss(I_low, I_high, hook=hook)
        loss_recon = F.l1_loss(I_low, I_high)
        loss_adjust =  loss_recon + loss_grad
        return loss_adjust

class Illum_Custom_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_loss(self, low, high):
        x_loss = F.l1_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.l1_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def gamma_loss(self, I_standard, I_high):
        loss = F.l1_loss(I_high, I_standard)
        return loss

    def forward(self, I_low, I_high, I_standard):
        loss_gamma = self.gamma_loss(I_standard, I_high)
        loss_grad = self.grad_loss(I_low, I_high)
        loss_recon = F.l1_loss(I_low, I_high)
        loss_adjust = loss_gamma + loss_recon + loss_grad
        return loss_adjust

# class VGGLoss(nn.Module):
#     def __init__(self, n_layers=5):
#         super().__init__()
#
#         feature_layers = (2, 7, 12, 21, 30)
#         self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         vgg = torchvision.models.vgg19(pretrained=True).features
#
#         self.layers = nn.ModuleList()
#         prev_layer = 0
#         for next_layer in feature_layers[:n_layers]:
#             layers = nn.Sequential()
#             for layer in range(prev_layer, next_layer):
#                 layers.add_module(str(layer), vgg[layer])
#             self.layers.append(layers.to(self.device))
#             prev_layer = next_layer
#
#         for param in self.parameters():
#             param.requires_grad = False
#
#         self.criterion = nn.L1Loss().to(self.device)
#
#     def forward(self, source, target):
#         loss = 0
#         for layer, weight in zip(self.layers, self.weights):
#             source = layer(source)
#             with torch.no_grad():
#                 target = layer(target)
#             loss += weight * self.criterion(source, target)
#
#         return loss
#
# vg=VGGLoss()

class Restore_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    # def col_loss(self,low,high):
    #     l0=low[:, 0, :, :]
    #     l0_mean=torch.mean(l0)
    #     h0 = high[:, 0, :, :]
    #     h0_mean = torch.mean(h0)
    #     s0=l0_mean*l0_mean -h0_mean*h0_mean
    #
    #     l1 = low[:, 1, :, :]
    #     l1_mean = torch.mean(l1)
    #     h1 = high[:, 1, :, :]
    #     h1_mean = torch.mean(h1)
    #     s1 =l1_mean*l1_mean -h1_mean*h1_mean
    #
    #     l2 = low[:, 2, :, :]
    #     l2_mean = torch.mean(l2)
    #     h2 = high[:, 2, :, :]
    #     h2_mean = torch.mean(h2)
    #     s2 = torch.abs(l2_mean- h2_mean)
    #     s=s0+s1+s2
    #     return s
    def col_loss(self,low,high):
        l=torch.mean(low,(2,3))
        h=torch.mean(high,(2,3))
        # print(l.shape)
        mr,mg,mb=torch.split(l,1,dim=1)

        drg=torch.pow(mr-mg,2)
        drb=torch.pow(mr-mb,2)
        dgb=torch.pow(mb-mg,2)
        k=torch.pow(torch.pow(drg,2)+torch.pow(drb,2)+torch.pow(dgb,2),0.5)
        s=torch.sum(k)
        return s



    def grad_loss(self, low, high, hook=-1):
        x_loss = F.mse_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.mse_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def forward(self, R_low, R_high, hook=-1):
        # loss_grad = self.grad_loss(R_low, R_high, hook=hook)

        # loss_col=self.col_loss(R_low,R_high)
        # loss_ssim = 1-self.ssim_loss(R_low, R_high)

        # y=100*loss_col
        # ss=1*loss_ssim
        # loss_recon = (F.l1_loss(R_low, R_high))*5
        # L = vg16(R_low,R_high)
        # vgloss = torch.mean(L)
        # for i in range(16):
        #     L.append(vg16(R_low[i, :, :, :], R_high[i, :, :, :]))
        # ls=F.smooth_l1_loss(R_low,R_high)
        # ls=ls*100
        vg19loss=(vg19(R_low,R_high))
        # vg19loss=vg19loss
        # l2 = (F.mse_loss(R_low, R_high)) * 5
        reuco=vg19loss

        # print('recon',loss_recon)
        print('vg19', vg19loss)
        # print('ss',ss)
        # # print('ssim',ss)
        #
        # loss_restore =rec+vgloss
        # print('vg',vgloss)
        # # print('ssim',ss)
        # print('recon',rec)
        return reuco


if __name__ == "__main__":
    from dataloader import *
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    Batch_size = 1
    log("Buliding LOL Dataset...")
    dst_test = LOLDataset(root_path_train, list_path_train, to_RAM=True, training=False)
    # But when we are training a model, the mean should have another value
    testloader = DataLoader(dst_test, batch_size = Batch_size)
    for i, data in enumerate(testloader):
        L_low, L_high, name = data
        L_gradient_x = gradient_no_abs(L_high, "x", device='cpu', kernel='sobel')
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        imgs = Denominator_x
        img = imgs[1].numpy()
        sample(img, figure_size=(1,1), img_dim=400)