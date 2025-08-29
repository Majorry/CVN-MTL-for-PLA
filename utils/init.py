import os
import random
import thop
import torch
from .model import get_model
from utils import logger, line_seg
# 这里的__all__是方面 import *
__all__ = ["init_device", "init_model"]

def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    # affinity表示的是亲和系数，os.getpid是获得当前进程号
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    # 种子存在random，torch设置随机种子
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Set the GPU id you choose
    """
    通过设置环境变量 CUDA_VISIBLE_DEVICES，指定要使用的 GPU。
    假设有多个 GPU 时，通过设置该变量可以限制程序只使用某一个 GPU（gpu 为指定的 GPU ID）
    """
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    """
    如果CPU不在用，GPU在用
    开启benchmark加速，如果存在种子就设置cuda一个种子
    pin_memory加速
    输出日志，显示当前程序正在使用的 GPU。如果没有明确指定 GPU，默认使用 GPU 0。
    """
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading

    model = get_model(args.model, args.user)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting

    # image = torch.randn([1, 1, 89, 92])
    """
    使用 thop 库的 profile 函数来计算模型的 FLOPs（浮点运算次数）和参数数量。
    inputs=(image,) 将输入图像传递给模型。
    """
    # verbose=False：设置为 False 时，不会在控制台中输出详细的计算过程信息。将其设置为 True 可以打印出每一层的计算情况。
    # Python 语法和函数参数解包机制有关。
    # flops, params = thop.profile(model, inputs=(image,), verbose=False)
    # 单位更易读
    # flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    # 打印日志
    logger.info(f'=> Model Name: {args.model} [pretrained: {args.pretrained}]')
    # logger.info(f'=> Model Flops: {flops}')
    # logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
