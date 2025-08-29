import argparse

__all__ = ['parse_args']

def parse_args():

    parser = argparse.ArgumentParser(description='CSI project PyTorch Training')
    # -----------------------------file name-----------------------
    parser.add_argument('--shape', default='transform', type=str, choices=['reshape', 'transform'])
    parser.add_argument('--user', type=int, default=8, help="Number of classes")
    parser.add_argument('--data_split', default='random', type=str, choices=['random', 'sequence', '2_1','3_1', '4_1', '5_1', '6_1'])
    parser.add_argument('--range', type=str, choices=['OATS_400', 'OATS_100', 'OATS_50', 'Simulate_0dB_300'], default='OATS_400',
                        help='用户划分范围')
    # -------------------------------------------------------------
    parser.add_argument('-b', '--batch_size', default=16, type=int, help="Batch size for training")
    parser.add_argument('-e', '--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--model', default='mtl_fc', choices=['mtl_fc', 'mtl_cvf' ,'mtl_conv1', 'CVN-MTL', 'mtl_cvc2'], help="Choose the model")
    parser.add_argument('--scheduler', type=str, default='const', choices=['cosine', 'const'], help="Choose the scheduler")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--save_pt', action='store_true', default=False, help="Save best model")
    #-------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--optimize_hparams",
        action="store_true",
        help="Whether to optimize hyperparameters with Optuna"
    )

    parser.add_argument('--pretrained', type=str, default=None,
                        help='using locally pre-trained model. The path of pre-trained model should be given')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test dataset')
    parser.add_argument('-w', '--num_workers', type=int, metavar='N', required=False, default=1,
                        help='number of data loading workers')
    parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--pin_memory', action='store_true', default = False, help='Enable pin memory for DataLoader')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--cpu', action='store_true',
                        help='disable GPU training (default: False)')
    parser.add_argument('--cpu-affinity', default=None, type=str,
                        help='CPU affinity, like "0xffff"')

    return parser.parse_args()

"python run_main.py --classes=8 -b 32 -e 50"