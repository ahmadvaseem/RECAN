from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return RECAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


## SFT Layer Group (SFT)
class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10, reduction=16):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, nf // reduction, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(nf // reduction, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, nf // reduction, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(nf // reduction, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        mm = feature_maps * mul
        mm = mm + add
        return mm

## Residual Block of SFT  (RB-SFT)
class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=10, reduction=16):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(nf=nf, para=para, reduction=reduction)
        self.sft2 = SFT_Layer(nf=nf, para=para, reduction=reduction)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.num_blocks = n_resblocks
        ##New Addition for SFT Block
        for i in range(n_resblocks):
            self.add_module('RCAB-residual' + str(i + 1), RCAB(conv=conv, n_feat=n_feat, kernel_size=kernel_size, reduction=reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
            self.add_module('SFT-residual' + str(i + 1), SFT_Residual_Block(nf=n_feat, para=1, reduction=reduction))

        self.groupConv = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        
 
        #modules_body = []
        
        #modules_body = [
        #    RCAB(
        #        conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
        #    SFT_Residual_Block(
        #        n_feat, 10) \
        #    for _ in range(n_resblocks)]
        
        #sft_branch = []
        #rcab_branch = []
        #for i in range(n_resblocks):
        #    sft_branch.append(SFT_Residual_Block())
        #    rcab_branch.append(RCAB())
        #self.sft_branch = nn.Sequential(*sft_branch)
        #self.rcab_branch = nn.Sequential(*rcab_branch)


#modules_body.append(conv(n_feat, n_feat, kernel_size))
        #self.body = nn.Sequential(*modules_body)

    def forward(self, input, ker_code):

        fea_in = input
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('RCAB-residual' + str(i + 1))(fea_in)
            fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, ker_code)
        
        fea_in = self.groupConv(fea_in)
        fea_add = torch.add(fea_in, input)
        #res = self.body(input)
        #res += fea_in
        return fea_add #res

## Residual Channel Attention Network (RECAN)
class RECAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RECAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        self.n_resgroups = n_resgroups
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # define body module
        #modules_body = [
        #    ResidualGroup(
        #        conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #    for _ in range(n_resgroups)]
        for i in range(n_resgroups):
            self.add_module('ResidualGroup' + str(i + 1), ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))
        
        self.groupConv0 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        #modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        
        #self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        

    def forward(self, input, ker_code):
        
        ## Blur Kernel Addition
        B, C, H, W = input.size() # I_LR batch
        #B_h, C_h, hx, bx = ker_code.size() # Batch, Len=10    
        #ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W)) #kernel_map stretch
        ker_code_exp = ker_code

        input = self.sub_mean(input)
        input = self.head(input)
        fea_in = input
        for i in range(self.n_resgroups):
            fea_in = self.__getattr__('ResidualGroup' + str(i + 1))(fea_in, ker_code_exp)

        res = fea_in #self.body(input)
        res = self.groupConv0(res)
        res += input

        input = self.tail(res)
        x = self.add_mean(input)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))