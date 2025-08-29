import math
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['WarmUpCosineAnnealingLR', 'FakeLR']

"""

定义一个增加和降低学习率的方法，一是通过不断增加学习率，二是通过余弦函数逐渐减少到最小学习率 （余弦退火阶段）

"""
class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    """
    self.base_lrs 是 _LRScheduler 类中的一个属性，它用于存储优化器中每个参数组的初始学习率。
    """
    def get_lr(self):
        # 这是不断增加学习率的条件判断
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        # 这里是余弦降低学习率的方法，方便收敛
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

# 返回的是初始学习率
class FakeLR(_LRScheduler):
    def __init__(self, optimizer):
        super(FakeLR, self).__init__(optimizer=optimizer)

    def get_lr(self):
        return self.base_lrs
