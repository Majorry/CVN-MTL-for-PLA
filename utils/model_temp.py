import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch
import complexPyTorch
import torch.nn.init as init
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
__all__ = ['get_model']
#--------------------------------------------Created Model--------------------------------------------------------------
class MultiTaskLearning_FC(nn.Module):
    def __init__(self, num_classes, input_shape=(2, 4, 2047)):
        super(MultiTaskLearning_FC, self).__init__()
        # 计算展平后的特征数
        self.flattened_size = input_shape[0] * input_shape[1] * input_shape[2]
        # 共享全连接特征提取部分
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),  # 输入尺寸需要是 flattened_size
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

        # 分类任务分支
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

        # 定位任务分支（回归）
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)  # 输出 x, y 坐标
        )

    def forward(self, x):
        # 将输入展平, 输入格式需为 [batch_size, channels, height, width]
        x = x.view(x.size(0), -1)
        # 共享全连接层提取特征
        shared_features = self.shared_fc(x)
        # 分别计算分类和定位的输出
        class_out = self.classifier(shared_features)
        loc_out = self.regressor(shared_features)
        return class_out, loc_out

#----------------------------------------------------------------------------------------------------------

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()

        self.real_linear = nn.Linear(in_features, out_features)
        self.imag_linear = nn.Linear(in_features, out_features)

    def forward(self, x):

        batch_size = x.size(0)
        real_part = x[:, :x.size(1) // 2]
        imag_part = x[:, x.size(1) // 2:]

        # 复数线性变换: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_output = self.real_linear(real_part) - self.imag_linear(imag_part)
        imag_output = self.real_linear(imag_part) + self.imag_linear(real_part)

        # 将实部和虚部连接起来
        return torch.cat([real_output, imag_output], dim=1)


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):

        real_part = x[:, :x.size(1) // 2]
        imag_part = x[:, x.size(1) // 2:]

        real_output = F.leaky_relu(real_part, negative_slope=self.negative_slope)
        imag_output = F.leaky_relu(imag_part, negative_slope=self.negative_slope)

        return torch.cat([real_output, imag_output], dim=1)


class ComplexMultiTaskLearning_FC(nn.Module):
    def __init__(self, num_classes, input_shape=(2, 4, 2047)):
        super(ComplexMultiTaskLearning_FC, self).__init__()

        self.flattened_size =  input_shape[1] * input_shape[2]


        self.shared_fc = nn.Sequential(
            ComplexLinear(self.flattened_size, 1024),
            ComplexLeakyReLU(),
            ComplexLinear(1024, 512),
            ComplexLeakyReLU(),
            ComplexLinear(512, 256),
            ComplexLeakyReLU()
        )


        self.classifier = nn.Sequential(
            ComplexLinear(256, 128),
            ComplexLeakyReLU(),
            nn.Linear(128 * 2, num_classes)  # 输入是复数特征(实部和虚部)，所以乘以2
        )


        self.regressor = nn.Sequential(
            ComplexLinear(256, 128),
            ComplexLeakyReLU(),
            nn.Linear(128 * 2, 2)  # 输出x,y坐标（实值）
        )

    def forward(self, x):
        # 将输入展平，假设输入格式为 [batch_size, channels, height, width]
        x = x.view(x.size(0), -1)

        shared_features = self.shared_fc(x)
        # 分别计算分类和定位的输出
        class_out = self.classifier(shared_features)
        loc_out = self.regressor(shared_features)
        return class_out, loc_out

#----------------------------------------------------------------------------------------------------------

class MultiTaskLearning_Conv1(nn.Module):
    def __init__(self, num_classes, input_shape=(2, 89, 92)):
        super(MultiTaskLearning_Conv1, self).__init__()
        # 共享特征提取部分 (简单 CNN)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),  # 根据输入通道数自动调整
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        # 使用一个虚拟输入计算经过共享卷积后展平的特征数
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # 生成一个虚拟输入
            conv_out = self.shared_conv(dummy_input)
            self.flattened_size = conv_out.view(1, -1).size(1)  # 动态计算展平后特征数量
        # 分类任务分支
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
        # 定位任务分支（回归）
        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)  # 输出 x, y 坐标
        )
    def forward(self, x):
        x = self.shared_conv(x)  # 提取共享特征
        x = self.flatten(x)      # 展平
        class_out = self.classifier(x)  # 分类输出
        loc_out = self.regressor(x)     # 定位输出
        return class_out, loc_out


#-----------------------------------------------------------------------------------------------------------------------
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        # 实部卷积与虚部卷积
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        # x: (B, 2C_in, H, W) → 拆分实部、虚部
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        # 复卷积运算
        y_re = self.conv_re(x_re) - self.conv_im(x_im)
        y_im = self.conv_re(x_im) + self.conv_im(x_re)
        # 拼接回 (B, 2C_out, H', W')
        return torch.cat([y_re, y_im], dim=1)

class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        return torch.cat([self.lrelu(x_re), self.lrelu(x_im)], dim=1)

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        return torch.cat([self.pool(x_re), self.pool(x_im)], dim=1)

class MultiTaskComplexCNN1(nn.Module):
    def __init__(self, num_classes, input_shape=(2, 89, 92)):
        super().__init__()
        in_channels = input_shape[0] // 2  # 复通道数
        # 1. 共享复数卷积特征提取
        self.shared = nn.Sequential(
            ComplexConv2d(in_channels, 8, kernel_size=3, padding=1),
            ComplexLeakyReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(8, 16, kernel_size=3, padding=1),
            ComplexLeakyReLU(),
            ComplexMaxPool2d(2, 2)
        )
        # 2. 计算展平后实值特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feats = self.shared(dummy)  # 复特征
            # 转幅度
            C2 = feats.size(1) // 2
            mag = torch.sqrt(feats[:, :C2] ** 2 + feats[:, C2:] ** 2)
            flat_size = mag.view(1, -1).size(1)
        # 3. 分类分支（全连接）
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
        # 4. 回归分支（全连接）
        self.regressor = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (B, 2C, H, W)
        z = self.shared(x)  # 复特征
        C2 = z.size(1) // 2
        mag = torch.sqrt(z[:, :C2] ** 2 + z[:, C2:] ** 2)  # 幅度
        feat = mag.view(mag.size(0), -1)  # 展平为 (B, D)
        class_out = self.classifier(feat)
        loc_out = self.regressor(feat)
        return class_out, loc_out


#-----------------------------------------------------------------

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        # 实部卷积与虚部卷积
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        # x: (B, 2C_in, H, W) → 拆分实部、虚部
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        # 复卷积运算
        y_re = self.conv_re(x_re) - self.conv_im(x_im)
        y_im = self.conv_re(x_im) + self.conv_im(x_re)
        # 拼接回 (B, 2C_out, H', W')
        return torch.cat([y_re, y_im], dim=1)

class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        return torch.cat([self.lrelu(x_re), self.lrelu(x_im)], dim=1)

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        C = x.size(1) // 2
        x_re, x_im = x[:, :C], x[:, C:]
        return torch.cat([self.pool(x_re), self.pool(x_im)], dim=1)

class MultiTaskComplexCNN1(nn.Module):
    def __init__(self, num_classes, input_shape=(2, 89, 92)):
        super().__init__()
        in_channels = input_shape[0] // 2  # 复通道数
        # 1. 共享复数卷积特征提取
        self.shared = nn.Sequential(
            ComplexConv2d(in_channels, 8, kernel_size=3, padding=1),
            ComplexLeakyReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(8, 16, kernel_size=3, padding=1),
            ComplexLeakyReLU(),
            ComplexMaxPool2d(2, 2)
        )
        # 2. 计算展平后实值特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feats = self.shared(dummy)  # 复特征
            # 转幅度
            C2 = feats.size(1) // 2
            mag = torch.sqrt(feats[:, :C2] ** 2 + feats[:, C2:] ** 2)
            flat_size = mag.view(1, -1).size(1)
        # 3. 分类分支（全连接）
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
        # 4. 回归分支（全连接）
        self.regressor = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (B, 2C, H, W)
        z = self.shared(x)  # 复特征
        C2 = z.size(1) // 2
        mag = torch.sqrt(z[:, :C2] ** 2 + z[:, C2:] ** 2)  # 幅度
        feat = mag.view(mag.size(0), -1)  # 展平为 (B, D)
        class_out = self.classifier(feat)
        loc_out = self.regressor(feat)
        return class_out, loc_out



def get_model(model_name, num_classes):
    model_dict = {
        'mtl_fc': MultiTaskLearning_FC,
        'mtl_cvf': ComplexMultiTaskLearning_FC,
        'mtl_conv1': MultiTaskLearning_Conv1,
        'mtl_cvc1': MultiTaskComplexCNN1
    }
    if model_name not in model_dict:
        raise ValueError("Unsupported model choice!")
    return model_dict[model_name](num_classes)
