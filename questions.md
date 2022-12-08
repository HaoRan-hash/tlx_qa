### 1. 问题1的标题
使用文字、代码块、图片描述问题1.

### 2. yolov5运行中报错

    Traceback (most recent call last):
      File "/home/fengtl/shenzhj/TLXZoo-main/demo/vision/object_detection/yolov5/train.py", line 77, in <module>
        trainer.train(n_epoch=n_epoch, train_dataset=coco.train, test_dataset=coco.test, print_freq=1,
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/model/core.py", line 114, in train
        self.th_train(
      File "/home/fengtl/shenzhj/TLXZoo-main/demo/vision/object_detection/yolov5/train.py", line 26, in th_train
        output = network(X_batch)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/shenzhj/TLXZoo-main/tlxzoo/vision/object_detection.py", line 26, in forward
        return self.backbone(inputs, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/shenzhj/TLXZoo-main/tlxzoo/module/yolov5/yolov5.py", line 70, in forward
        feat1, feat2, feat3 = self.backbone(x)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/shenzhj/TLXZoo-main/tlxzoo/module/yolov5/CSPdarknet.py", line 156, in forward
        x = self.stem(x)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/shenzhj/TLXZoo-main/tlxzoo/module/yolov5/CSPdarknet.py", line 34, in forward
        return self.conv(tlx.ops.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/shenzhj/TLXZoo-main/tlxzoo/module/yolov5/CSPdarknet.py", line 52, in forward
        return self.act(self.bn(self.conv(x)))
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/core/core_torch.py", line 123, in _call_impl_tlx
        result = self._call_impl(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/nn/layers/convolution/simplified_conv.py", line 293, in forward
        outputs = self.conv2d(inputs, self.filters)
      File "/home/fengtl/miniconda3/envs/py39torch110/lib/python3.9/site-packages/tensorlayerx/backend/ops/torch_nn.py", line 594, in __call__
        output = F.conv2d(input, filters, stride=self.strides, padding=self.padding,
    TypeError: conv2d() received an invalid combination of arguments - got (Tensor, Parameter, groups=int, stride=tuple, dilation=tuple, padding=tuple), but expected one of:
     * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)
     * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, str padding, tuple of ints dilation, int groups)


    Process finished with exit code 1


**这是CSPdarknet.py**

```python
import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
class SiLU(nn.Module):
    #@staticmethod
    def forward(x):
        # return x * torch.sigmoid(x)
        return x * tlx.sigmoid(x)
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, act)
    def forward(self, x):
        # 按维数1拼接（横着拼）
        #return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        #print(x[..., ::2, ::2].shape) [128, 640, 320, 2]
        #print( x[..., 1::2, ::2].shape) [128, 640, 320, 2]
        #print(x[..., ::2, 1::2].shape) [128, 640, 320, 1]
        #print(x[..., 1::2, 1::2].shape) [128, 640, 320, 1]
        return self.conv(tlx.ops.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Conv, self).__init__()
        """
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        """
        self.conv = nn.Conv2d(out_channels=c2, kernel_size=(k, k), stride=(s, s), padding=(p, p), b_init=None,
                              in_channels=c1)
        self.bn = nn.BatchNorm2d(momentum=0.03, epsilon=0.001, num_features=c2)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        #print(x.type)
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self, x):
        return self.act(self.conv(x))
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        layer_list = []
        for i in range(n):
            layer_list.append(Bottleneck(c_, c_, shortcut, g, e=1.0))
        self.m = nn.Sequential(layer_list)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        ## self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.cv3(tlx.ops.concat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    def forward(self, x):
        x = self.cv1(x)
        # return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return self.cv2(tlx.ops.concat([x] + [m(x) for m in self.m], 1))
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, k=3)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
```
