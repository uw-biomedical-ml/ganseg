from GANSeg import GANSeg
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of GANSeg. The GAN part uses U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test / test_ukb / test_eyeact]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='heidelberg_to_topcon', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')
    parser.add_argument('--seg_weight', type=int, default=1000, help='Weight for Segmenter')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=1, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    ## GANSeg options
    # data options
    parser.add_argument('--seg_classes', type=int, default=8, help='Number of segmentation classes')
    parser.add_argument('--class_weight_file', type=str, default=None, help='path to class weights')
    parser.add_argument('--aug_options_file', type=str, default="aug_options.json", help='path to augmentation options')
    parser.add_argument('--seg_visual_factor', type=int, default=30, help='to help visualise seg masks')
    # model options
    parser.add_argument('--no_gan', type=bool, default=False, help='GAN or just UNet')
    parser.add_argument('--no_seg', type=bool, default=False, help='Learn Seg or Not')
    parser.add_argument('--add_seg_link', type=bool, default=False, help='Seg link loss on A, A2B etc')
    parser.add_argument('--U_A2B2A', type=bool, default=False, help='Segment A2B2A')
    parser.add_argument('--seg_loss', type=str, default='NLL', help='segmentation loss function')

    parser.add_argument('--testB_folder', type=str, default="testB")
    parser.add_argument('--test_start_index', type=int, default=0)
    parser.add_argument('--test_end_index', type=int, default=-1)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = GANSeg(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

    if args.phase == 'test_ukb':
        gan.test_ukb()
        print(" [*] Test finished!")

    if args.phase=="test_eyeact":
        gan.test_eyeact()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
