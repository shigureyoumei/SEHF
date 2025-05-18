import os
import logging

import torch
import torch.nn as nn
import sys
import model
import model.Generator

BN_MOMENTUM = 0.9
sys.path.append("..")


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """
	#只需要输入输入的数据的通道，因为后续的通道数不会发生改变
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input（输入的图像的最后一个维度，图像的通道数）
        * `n_heads` is the number of heads in multi-head attention（多头注意力的头的个数）
        * `d_k` is the number of dimensions in each head（每个多头注意力的维度数）
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
       
        #组归一化：你有一个包含64个通道的输入，并且你设置n_groups=8，那么每个组将包含8个通道，组归一化将在这8个通道上独立地计算均值和标准差，并进行归一化
        self.norm = nn.GroupNorm(n_groups, n_channels)
        
        #将通过线性变化，将通道数增大为：多头注意力头数*每个头的维度，以及*3，用来后续划分为Q、K、V
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
     
        #输出为维度通过线性映射恢复为何输入一致
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        
  
        #首先得到输入数据的批量大小，通道维度，长，宽
        batch_size, n_channels, height, width = x.shape
       
        #将除通道数的维度进行合并，然后将通道数放在最后面
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
    
        #通过投影，将数据的维度进行提升，让其满足多头注意力的维度数
        #然后将数据的维度变化为：批量大小，像素维度（比如长*宽），头数，3*头的维度
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
   
        #将得到QKV按照最后一个维度进行维度划分，得到QKV矩阵
        q, k, v = torch.chunk(qkv, 3, dim=-1)
  
        #QK进行点积计算，维度变为：批量，像素维度，像素维度，头数
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        #将第二个维度归一化
        attn = attn.softmax(dim=2)
        #attn与v进行点积，实现加权计算，维度变为和输入的QKV一样：批量，像素大小，头数，每头维度
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        #将结果的最后两个维度合并：头数，像素大小，升维的维度
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        #将结果的维度调整为和输入一致
        res = self.output(res)
		#做残差连接
        res += x
		#将将结果的维度调整为和输入一致，将长和宽拆开
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res




class PoseResNet(nn.Module):
    def __init__(self, block, layers,  **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.temporal = 4

        # self._myparams = nn.Parameter(torch.stack([torch.randn(self.CFG.HEATMAP_SIZE) for i in
        #                                            range((self.CFG.temporal * (self.CFG.temporal - 1)) // 2)]).unsqueeze(dim=0))
        self._myparams = nn.Parameter(torch.stack([torch.randn([280, 448]) for i in
                                                   range((4 * (4 - 1)) // 2)]).unsqueeze(dim=0))

        # lstm
        # self.outclass = CFG.NUM_JOINTS
        self.conv_ix_lstm = nn.Conv2d(256+3, 256, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(256+3, 256, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(256+3, 256, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(256+3, 256, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # initial lstm
        self.conv_gx_lstm0 = nn.Conv2d(256+3, 256, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(256+3, 256, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(256+3, 256, kernel_size=3, padding=1)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
            in_channels=256,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0)
        )

        self.generator = model.Generator.get_generator()
        self.attn = AttentionBlock(4, n_heads=4, d_k=4, n_groups=2)
        self.RGBprelayer = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.eventprelayer = nn.Conv2d(2, 4, kernel_size=3, padding=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def lstm(self, heatmap, features, hide_t_1, cell_t_1):
        '''
        :param heatmap:     (class+1) * 45 * 45
        :param features:    32 * 45 * 45
        :param centermap:   1 * 45 * 45
        :param hide_t_1:    48 * 45 * 45
        :param cell_t_1:    48 * 45 * 45
        :return:
        hide_t:    48 * 45 * 45
        cell_t:    48 * 45 * 45
        '''
        xt = torch.cat([heatmap, features], dim=1)  # (32+ class+1 +1 ) * 45 * 45

        gx = self.conv_gx_lstm(xt)  # output: 48 * 45 * 45
        gh = self.conv_gh_lstm(hide_t_1)  # output: 48 * 45 * 45
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)  # output: 48 * 45 * 45
        oh = self.conv_oh_lstm(hide_t_1)  # output: 48 * 45 * 45
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)  # output: 48 * 45 * 45
        ih = self.conv_ih_lstm(hide_t_1)  # output: 48 * 45 * 45
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)  # output: 48 * 45 * 45
        fh = self.conv_fh_lstm(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)

        return cell_t, hide_t

    def lstm0(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

    def _resnet2(self, x):  # ConvNet1
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.deconv_layers(x)
        x = self.generator(x)

        return x

    def _resnet3(self, x):
        x = self.final_layer(x)
        return x #3*280*448
    


    def forward(self, events, rgb):
        heat_maps = []
        event0 = events[:, 0, :, :, :]
        '这里加入rgb输入'
        rgb = self.RGBprelayer(rgb)
        event0 = self.eventprelayer(event0)
        event0 = rgb + event0
        # event0 = self.attn(event0)
        initial_heatmap = self._resnet3(self._resnet2(event0))
        feture = self._resnet2(event0)
        x = torch.cat([initial_heatmap, feture], dim=1)
        cell, hide = self.lstm0(x)
        heatmap = self._resnet3(hide)
        heat_maps.append(heatmap)
        num_heat = 0
        for i in range(1, 4):
            heatmap_new = torch.zeros(heatmap.size()).cuda()
            for j in range(i):
                heatmap_new = torch.mul(heat_maps[j], self._myparams[:, num_heat]) + heatmap_new
                num_heat += 1
            heatmap = heatmap_new
            event = events[:, i]
            event = self.eventprelayer(event)
            # event = self.attn(event)
            feature = self._resnet2(event)
            cell, hide = self.lstm(heatmap, feature, hide, cell)
            heatmap = self._resnet3(hide)
            heat_maps.append(heatmap)
        return torch.stack(heat_maps, dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3])
               # ,
               #            50: (Bottleneck, [3, 4, 6, 3]),
               #            101: (Bottleneck, [3, 4, 23, 3]),
               #            152: (Bottleneck, [3, 8, 36, 3])
               }


def get_pose_net():
    num_layers = 18
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers)
    if 1 and 1:
        model.init_weights()
    return model