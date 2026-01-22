import time, itertools, json
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import torch.nn.functional as F


class GANSeg(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.seg_weight = args.seg_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_ch = args.img_ch
        self.img_size = args.img_size
        self.col_size = args.img_size

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        """ GANSeg options """
        """ GANSeg data options """
        self.class_weight_file = args.class_weight_file
        self.aug_options_file = args.aug_options_file
        self.seg_classes = args.seg_classes
        self.seg_visual_factor = args.seg_visual_factor

        """ GANSeg model options """
        self.no_gan = args.no_gan
        self.no_seg = args.no_seg
        self.add_seg_link = args.add_seg_link
        self.seg_loss = args.seg_loss
        self.U_A2B2A = args.U_A2B2A

        """ Test Options """
        self.testB_folder = args.testB_folder
        self.test_start_index = args.test_start_index
        self.test_end_index = args.test_end_index

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        args_json_path = os.path.join(self.result_dir, self.dataset, "config.json")
        with open(args_json_path, 'w') as fout:
            json.dump(args.__dict__, fout)
        fout.close()

        print()
        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)
        print("# seg_weight : ", self.seg_weight)

        print("##### GANSeg options #####")
        print("# no_seg (pure U-GAT-IT) : ", self.no_seg)
        print("# seg_classes : ", self.seg_classes)
        print("# add_seg_link (loss between segs?) : ", self.add_seg_link)
        print("# seg_factor (how to adjust seg masks) : ", self.seg_visual_factor)
        print("# class_weight_file : ", self.class_weight_file)
        print("# aug_options_file : ", self.aug_options_file)
        print("# seg_loss : ", self.seg_loss)
        print("# testB_folder : ", self.testB_folder)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        aug_options_path = os.path.join('dataset', self.dataset, self.aug_options_file)
        aug_options = json.loads(open(aug_options_path).read())

        """ DataLoader """
        if "normalize" in aug_options:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*self.img_ch, std=[0.5]*self.img_ch),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*self.img_ch, std=[0.5]*self.img_ch),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        aug_options_resize = {"resize":{"probability":1, "width":self.img_size, "height":self.img_size}}
        print('train_transform', train_transform, 'test_transform', test_transform)
        print('aug_options', aug_options, 'aug_options_resize', aug_options_resize)

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), transform=train_transform,
                                  img_size=self.img_size, num_classes=self.seg_classes, num_ch=self.img_ch,
                                  seg_factor=self.seg_visual_factor, aug_options=aug_options, col_size=self.col_size)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), transform=train_transform,
                                  img_size=self.img_size, num_classes=self.seg_classes, num_ch=self.img_ch,
                                  seg_factor=self.seg_visual_factor, aug_options=aug_options, col_size=self.col_size)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'validA'), transform=test_transform,
                                 img_size=self.img_size, num_classes=self.seg_classes, num_ch=self.img_ch,
                                 seg_factor=self.seg_visual_factor, aug_options=aug_options_resize, col_size=self.col_size)
        self.realtestA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), transform=test_transform,
                                     img_size=self.img_size, num_classes=self.seg_classes, num_ch=self.img_ch,
                                     seg_factor=self.seg_visual_factor, aug_options=aug_options_resize, col_size=self.col_size)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, self.testB_folder), transform=test_transform,
                                 img_size=self.img_size, num_classes=self.seg_classes, num_ch=self.img_ch,
                                 seg_factor=self.seg_visual_factor, aug_options=aug_options_resize, col_size=self.col_size)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=False)
        self.realtestA_loader = DataLoader(self.realtestA, batch_size=self.batch_size, shuffle=False)
        self.testA_loader = DataLoader(self.testA, batch_size=self.batch_size, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=self.batch_size, shuffle=False)

        """ Define Generator, Discriminator """
        if not self.no_gan:
            self.genA2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
            self.genB2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
            num_adj = int(256/self.img_size/2)
            self.disGA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=(7-num_adj)).to(self.device)
            self.disGB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=(7-num_adj)).to(self.device)
            self.disLA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=(5-num_adj)).to(self.device)
            self.disLB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=(5-num_adj)).to(self.device)

        self.seg = UnetGenerator(input_nc=self.img_ch, output_nc=self.seg_classes, num_downs=8, ngf=self.ch).to(self.device)  #shared

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        if self.seg_loss=='NLL':
            if self.class_weight_file:
                class_weight_path = os.path.join('dataset', self.dataset, self.class_weight_file)
                class_weights_seg = np.loadtxt(class_weight_path).astype(np.float32)
                class_weights_seg = torch.from_numpy(class_weights_seg).to(self.device)
                self.NLL_loss = nn.NLLLoss(weight=class_weights_seg).to(self.device)
            else:
                self.NLL_loss = nn.NLLLoss().to(self.device)
        elif self.seg_loss=='focal':
            from focalLoss import FocalLoss
            self.NLL_loss = FocalLoss(gamma=0.5, alpha=None).to(self.device)
        elif self.seg_loss == 'normFocal':
            from focalLoss import NormalizedFocalLoss
            self.NLL_loss = NormalizedFocalLoss(gamma=0.5, alpha=None, num_classes=self.seg_classes).to(self.device)
        elif self.seg_loss == 'normFocalWeighted':
            from focalLoss import NormalizedFocalLossWeighted
            class_weight_path = os.path.join('dataset', self.dataset, self.class_weight_file)
            class_weights_seg = np.loadtxt(class_weight_path).astype(np.float32)
            class_weights_seg = torch.from_numpy(class_weights_seg).to(self.device)
            self.NLL_loss = NormalizedFocalLossWeighted(gamma=0.5, alpha=None, num_classes=self.seg_classes, weights=class_weights_seg).to(self.device)

        """ Trainer """
        if not self.no_gan:
            self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                                                            self.disLA.parameters(), self.disLB.parameters()),
                                            lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        if self.no_seg:     # segmentation not backpropped
            self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                            lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        else:
            if self.no_gan:
                self.G_optim = torch.optim.Adam(itertools.chain(self.seg.parameters()), lr=self.lr, betas=(0.5, 0.999),
                                                weight_decay=self.weight_decay)
            else:
                # chain unet to generator
                self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(),
                                                                self.genB2A.parameters(),
                                                                self.seg.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        if not self.no_gan:
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        self.seg.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                self.load(os.path.join(model_list[-1]))
                start_iter = model_list[-1].split('_')[-1].split('.')[0]
                if start_iter=='latest':
                    start_iter= 0
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):  # lr schedule
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    if not self.no_gan:
                        self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        num_trainA = len(self.trainA_loader)    # actually num_batches
        num_testA = len(self.testA_loader)
        num_trainB = len(self.trainB_loader)
        num_testB = len(self.testB_loader)

        print("num_trainA=", num_trainA, "num_trainB=", num_trainB, "num_testA=", num_testA, "num_testB=", num_testB)
        best_U_loss_valid = np.float("inf")

        valid_file = os.path.join(self.result_dir, self.dataset, 'valid_logs.csv')
        with open(valid_file, 'w') as fout:
            fout.write('epoch, U_loss_A, U_loss_A2B \n')
        fout.close()

        log_file = os.path.join(self.result_dir, self.dataset, 'train_logs.csv')
        with open(log_file, 'w') as fout:
            fout.write("iter, epoch, D_loss, D_loss_A, D_loss_B, "
                       "G_loss, G_loss_A, G_loss_B, "
                       "G_ad_loss_GA, G_ad_cam_loss_GA, G_ad_loss_LA," "G_ad_cam_loss_LA, G_recon_loss_A, G_identity_loss_A, G_cam_loss_A, "
                       "G_ad_loss_GB, G_ad_cam_loss_GB, G_ad_loss_LB," "G_ad_cam_loss_LB, G_recon_loss_B, G_identity_loss_B, G_cam_loss_B,"
                       "U_loss_A, U_loss_A2B\n")
        fout.close()

        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                if not self.no_gan:
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, real_seg_A = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, real_seg_A = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device, dtype=torch.float), real_B.to(self.device, dtype=torch.float)
            real_seg_A = real_seg_A.to(self.device, dtype=torch.long)

            if not self.no_gan:
                # Update D
                self.D_optim.zero_grad()

                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                iter_losses_D = [D_loss_A.detach().cpu().numpy(),
                                 D_loss_B.detach().cpu().numpy(),
                                 Discriminator_loss.detach().cpu().numpy()]

                Discriminator_loss.backward()
                self.D_optim.step()
            else:
                iter_losses_D = [0, 0, 0]

            # Update G
            self.G_optim.zero_grad()

            if not self.no_gan:
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

                G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) \
                            + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) \
                           + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                # # update U (segmenter)
                seg_A = self.seg(real_A)
                seg_A2B = self.seg(fake_A2B)

                if self.seg_loss=='maxSquare':
                    pred_P = F.softmax(seg_A, dim=1)
                    pred_P_A2B = F.softmax(seg_A2B, dim=1)
                    print(seg_A.shape, pred_P.shape, seg_A2B.shape, pred_P_A2B.shape)
                    U_loss_A = self.NLL_loss(seg_A, pred_P)
                    U_loss_A2B = self.NLL_loss(seg_A2B, pred_P)
                    U_loss_A_A2B = self.L1_loss(seg_A, pred_P_A2B)  # might be sharper?!
                else:
                    U_loss_A = self.NLL_loss(seg_A, real_seg_A)
                    U_loss_A2B = self.NLL_loss(seg_A2B, real_seg_A)
                    U_loss_A_A2B = self.L1_loss(seg_A, seg_A2B)    # might be sharper?!
                    if self.U_A2B2A:
                        seg_A2B2A = self.seg(fake_A2B2A)
                        U_loss_A2B2A = self.NLL_loss(seg_A2B2A, real_seg_A)

                if self.add_seg_link:
                    Segmenter_loss = self.seg_weight * (U_loss_A + U_loss_A2B + U_loss_A_A2B)
                else:
                    Segmenter_loss = self.seg_weight * (U_loss_A + U_loss_A2B)

                if self.U_A2B2A:
                    Segmenter_loss += self.seg_weight * (U_loss_A2B2A)
                    if self.add_seg_link:
                        U_loss_A_A2B2A = self.L1_loss(seg_A, seg_A2B2A)
                        U_loss_A2B_A2B2A = self.L1_loss(seg_A2B, seg_A2B2A)
                        Segmenter_loss += self.seg_weight * (U_loss_A_A2B2A + U_loss_A2B_A2B2A)

                if self.no_seg:
                    Generator_loss = G_loss_A + G_loss_B
                else:
                    Generator_loss = G_loss_A + G_loss_B + Segmenter_loss

                iter_losses_G = [
                                Generator_loss.detach().cpu().numpy(),
                                 G_loss_A.detach().cpu().numpy(),
                                 G_loss_B.detach().cpu().numpy(),
                                 G_ad_loss_GA.detach().cpu().numpy(),
                                 G_ad_cam_loss_GA.detach().cpu().numpy(),
                                 G_ad_loss_LA.detach().cpu().numpy(),
                                 G_ad_cam_loss_LA.detach().cpu().numpy(),
                                 G_recon_loss_A.detach().cpu().numpy(),
                                 G_identity_loss_A.detach().cpu().numpy(),
                                 G_cam_loss_A.detach().cpu().numpy(),
                                 G_ad_loss_GB.detach().cpu().numpy(),
                                 G_ad_cam_loss_GB.detach().cpu().numpy(),
                                 G_ad_loss_LB.detach().cpu().numpy(),
                                 G_ad_cam_loss_LB.detach().cpu().numpy(),
                                 G_recon_loss_B.detach().cpu().numpy(),
                                 G_identity_loss_B.detach().cpu().numpy(),
                                 G_cam_loss_B.detach().cpu().numpy(),
                                 U_loss_A.detach().cpu().numpy(),
                                 U_loss_A2B.detach().cpu().numpy()]
            else:   # stand alone unet
                seg_A = self.seg(real_A)
                U_loss_A = self.NLL_loss(seg_A, real_seg_A)
                Generator_loss = U_loss_A
                iter_losses_G = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U_loss_A.detach().cpu().numpy(), 0]

            with open(log_file, 'a') as fout:
                cur_epoch = step // num_trainA
                vals = [step, cur_epoch] + iter_losses_D + iter_losses_G
                fout.write("{}\n".format(",".join([str(x) for x in vals])))
            fout.close()

            Generator_loss.backward()
            self.G_optim.step()

            if not self.no_gan:
                # clip parameter of AdaILN and ILN, applied after optimizer step
                self.genA2B.apply(self.Rho_clipper)
                self.genB2A.apply(self.Rho_clipper)
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            else:
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, 0, Generator_loss))

            if step % (self.print_freq * num_trainA)==0:  # print by epoch
                cur_epoch = step // num_trainA

                test_sample_num = int(num_testA)     # do whole test (actually validation)
                train_sample_num = 3
                print('in validation loop; step=', step, "batch_size=", self.batch_size, "num_trainA:", num_trainA,
                      "cur_epoch=", cur_epoch, "num_testA", num_testA, "test_sample_num=", test_sample_num)

                acc = 0
                if self.no_gan:
                    self.seg.eval()

                    ## force everything to be 3 channels
                    if self.img_ch == 3:
                        SEG_A = np.zeros((self.img_size * 3, 0, 3))  # A_real, U(A)
                    else:
                        SEG_A = np.zeros((self.img_size * 3, 0))

                    with torch.no_grad():
                        SEG_A, _, _ = self.evaluate_seg_no_gan(train_sample_num, SEG_A, is_train=True)
                        SEG_A, U_loss_A, acc = self.evaluate_seg_no_gan(test_sample_num, SEG_A, is_train=False)

                    U_loss_A_np = tensor2numpy_v2(U_loss_A)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'SEGA_%07d_%.4f.png' % (step, U_loss_A_np)), SEG_A * self.seg_visual_factor)

                    with open(valid_file, 'a') as fout:
                        vals = [cur_epoch, U_loss_A_np, 0, acc]
                        fout.write("{}\n".format(",".join([str(x) for x in vals])))
                    fout.close()
                    if U_loss_A_np < best_U_loss_valid:
                        best_U_loss_valid = U_loss_A_np
                        save_name = self.dataset + '_params_best.pt'
                        self.save(os.path.join(self.result_dir, self.dataset, 'model'), savename=save_name)

                    self.seg.train()
                else:
                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    self.seg.eval()

                    ## force everything to be 3 channels
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))
                    if self.img_ch==3:
                        SEG_A = np.zeros((self.img_size * 5, 0, 3))
                    else:
                        SEG_A = np.zeros((self.img_size * 5, 0))

                    with torch.no_grad():
                        A2B, B2A, SEG_A, _, _, _, _ = self.evaluate_seg(train_sample_num, A2B, B2A, SEG_A, is_train=True)
                        A2B, B2A, SEG_A, U_loss_A, U_loss_B, U_loss_B_B2B, acc = self.evaluate_seg(test_sample_num, A2B, B2A, SEG_A, is_train=False)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)

                    U_loss_A_np = tensor2numpy_v2(U_loss_A)
                    U_loss_A2B_np = tensor2numpy_v2(U_loss_A2B)
                    if U_loss_B_B2B!=0:
                        U_loss_B_B2B_np = tensor2numpy_v2(U_loss_B_B2B)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'SEGA_%07d_%.4f_%.4f.png'
                                             % (step, U_loss_A_np, U_loss_A2B_np)), SEG_A *self.seg_visual_factor)
                    if not self.no_seg:
                        with open(valid_file, 'a') as fout:
                            vals = [cur_epoch, U_loss_A_np, U_loss_A2B_np, acc]
                            fout.write("{}\n".format(",".join([str(x) for x in vals])))
                        fout.close()

                        if U_loss_A_np + U_loss_A2B_np < best_U_loss_valid:
                            best_U_loss_valid = U_loss_A_np + U_loss_A2B_np
                            save_name = self.dataset + '_params_best.pt'
                            self.save(os.path.join(self.result_dir, self.dataset, 'model'), savename=save_name)

                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
                    if not self.no_seg:
                        self.seg.train()

            if step % self.save_freq == 0:
                save_name = self.dataset + '_params_%07d.pt' % step
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), savename=save_name)

            if step % 1000 == 0:
                save_name = self.dataset + '_params_latest.pt'
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), savename=save_name)


    def evaluate_seg_no_gan(self, num_samples, SEG_A, is_train=True):
        U_loss_A = 0
        for _ in range(num_samples):
            try:
                real_A, real_seg_A = A_iter.next()
            except:
                if is_train:
                    A_iter = iter(self.trainA_loader)
                else:
                    A_iter = iter(self.testA_loader)
                real_A, real_seg_A = A_iter.next()

            real_A = real_A.to(self.device, dtype=torch.float)
            real_seg_A = real_seg_A.to(self.device, dtype=torch.long)

            seg_A = self.seg(real_A)
            seg_A_thresh = np.argmax(tensor2numpy_v2(seg_A[0]), axis=0)
            real_seg_A_np = tensor2numpy_v2(real_seg_A[0])
            real_A_for_seg = (tensor2numpy(denorm(real_A[0]))[:, :, 0] * 255 / self.seg_visual_factor)
            if self.img_ch == 3:  # accommodate real_B
                real_A_for_seg = (tensor2numpy(denorm(real_A[0])) * 255 / self.seg_visual_factor)
                real_seg_A_np = np.repeat(np.expand_dims(real_seg_A_np, axis=2), 3, axis=2)
                seg_A_thresh = np.repeat(np.expand_dims(seg_A_thresh, axis=2), 3, axis=2)
            SEG_A = np.concatenate((SEG_A, np.concatenate((real_A_for_seg, real_seg_A_np, seg_A_thresh))), 1)
            U_loss_A += self.NLL_loss(seg_A, real_seg_A)
        return SEG_A, U_loss_A, 0


    def evaluate_seg(self, num_samples, A2B, B2A, SEG_A, is_train=True):
        U_loss_A, U_loss_A2B, U_loss_B_B2B = 0, 0, 0
        for _ in range(num_samples):
            try:
                real_A, real_seg_A = A_iter.next()
            except:
                if is_train:
                    A_iter = iter(self.trainA_loader)
                else:
                    A_iter = iter(self.testA_loader)
                real_A, real_seg_A = A_iter.next()

            try:
                real_B, _ = B_iter.next()
            except:
                if is_train:
                    B_iter = iter(self.trainB_loader)
                else:
                    B_iter = iter(self.testB_loader)
                real_B, _ = B_iter.next()

            real_A, real_B = real_A.to(self.device, dtype=torch.float), real_B.to(self.device, dtype=torch.float)
            real_seg_A = real_seg_A.to(self.device, dtype=torch.long)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            real_A_np = tensor2numpy(denorm(real_A[0]))
            fake_A2A_np = tensor2numpy(denorm(fake_A2A[0]))
            fake_A2B_np = tensor2numpy(denorm(fake_A2B[0]))
            fake_A2B2A_np = tensor2numpy(denorm(fake_A2B2A[0]))
            fake_A2A_heatmap_np = cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size)
            fake_A2B_heatmap_np = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
            fake_A2BA_heatmap_np = cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size)
            if self.img_ch != 3:
                real_A_np = np.repeat(real_A_np, 3, axis=2)
                fake_A2A_np = np.repeat(fake_A2A_np, 3, axis=2)
                fake_A2B_np = np.repeat(fake_A2B_np, 3, axis=2)
                fake_A2B2A_np = np.repeat(fake_A2B2A_np, 3, axis=2)
            A2B = np.concatenate((A2B, np.concatenate((real_A_np,
                                                       fake_A2A_heatmap_np,
                                                       fake_A2A_np,
                                                       fake_A2B_heatmap_np,
                                                       fake_A2B_np,
                                                       fake_A2BA_heatmap_np,
                                                       fake_A2B2A_np), 0)), 1)

            real_B_np = tensor2numpy(denorm(real_B[0]))
            fake_B2B_np = tensor2numpy(denorm(fake_B2B[0]))
            fake_B2A_np = tensor2numpy(denorm(fake_B2A[0]))
            fake_B2A2B_np = tensor2numpy(denorm(fake_B2A2B[0]))
            fake_B2B_heatmap_np = cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size)
            fake_B2A_heatmap_np = cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size)
            fake_B2A2B_heatmap_np = cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size)
            if self.img_ch != 3:
                real_B_np = np.repeat(real_B_np, 3, axis=2)
                fake_B2B_np = np.repeat(fake_B2B_np, 3, axis=2)
                fake_B2A_np = np.repeat(fake_B2A_np, 3, axis=2)
                fake_B2A2B_np = np.repeat(fake_B2A2B_np, 3, axis=2)
            B2A = np.concatenate((B2A, np.concatenate((real_B_np,
                                                       fake_B2B_heatmap_np,
                                                       fake_B2B_np,
                                                       fake_B2A_heatmap_np,
                                                       fake_B2A_np,
                                                       fake_B2A2B_heatmap_np,
                                                       fake_B2A2B_np), 0)), 1)

            seg_A = self.seg(real_A)
            seg_A2B = self.seg(fake_A2B)
            seg_B = self.seg(real_B)

            seg_A_thresh = np.argmax(tensor2numpy_v2(seg_A[0]), axis=0)
            seg_A2B_thresh = np.argmax(tensor2numpy_v2(seg_A2B[0]), axis=0)
            seg_B_thresh = np.argmax(tensor2numpy_v2(seg_B[0]), axis=0)
            real_seg_A_np = tensor2numpy_v2(real_seg_A[0])
            real_B_for_seg = (tensor2numpy(denorm(real_B[0]))[:, :, 0] * 255 / self.seg_visual_factor)
            if self.img_ch == 3:  # accommodate real_B
                real_B_for_seg = (tensor2numpy(denorm(real_B[0])) * 255 / self.seg_visual_factor)
                real_seg_A_np = np.repeat(np.expand_dims(real_seg_A_np, axis=2), 3, axis=2)
                seg_A_thresh = np.repeat(np.expand_dims(seg_A_thresh, axis=2), 3, axis=2)
                seg_A2B_thresh = np.repeat(np.expand_dims(seg_A2B_thresh, axis=2), 3, axis=2)
                seg_B_thresh = np.repeat(np.expand_dims(seg_B_thresh, axis=2), 3, axis=2)
            SEG_A = np.concatenate((SEG_A, np.concatenate((real_seg_A_np,
                                                           seg_A_thresh,
                                                           seg_A2B_thresh,
                                                           real_B_for_seg,
                                                           seg_B_thresh))), 1)
            U_loss_A += self.NLL_loss(seg_A, real_seg_A)
            U_loss_A2B += self.NLL_loss(seg_A2B, real_seg_A)

        return A2B, B2A, SEG_A, U_loss_A, U_loss_A2B, U_loss_B_B2B, 0


    def save(self, dir, savename):
        params = {}
        if not self.no_gan:
            params['genA2B'] = self.genA2B.state_dict()
            params['genB2A'] = self.genB2A.state_dict()
            params['disGA'] = self.disGA.state_dict()
            params['disGB'] = self.disGB.state_dict()
            params['disLA'] = self.disLA.state_dict()
            params['disLB'] = self.disLB.state_dict()
            params['seg'] = self.seg.state_dict()
        else:
            params['seg'] = self.seg.state_dict()
        torch.save(params, os.path.join(dir, savename))
        return


    def load(self, fn):
        params = torch.load(fn)
        if not self.no_gan:
            self.genA2B.load_state_dict(params['genA2B'])
            self.genB2A.load_state_dict(params['genB2A'])
            self.disGA.load_state_dict(params['disGA'])
            self.disGB.load_state_dict(params['disGB'])
            self.disLA.load_state_dict(params['disLA'])
            self.disLB.load_state_dict(params['disLB'])
            self.seg.load_state_dict(params['seg'])
        else:
            self.seg.load_state_dict(params['seg'])
        return


    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*best.pt'))
        print(model_list)
        if not len(model_list) == 0:
            self.load(model_list[0])
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        if not self.no_seg:
            self.seg.eval()
            out_dir = os.path.join(self.result_dir, self.dataset, self.testB_folder)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            do_train_B = 0
            loader = self.trainB_loader if do_train_B else self.realtestA_loader
            prefix = "trainB" if do_train_B else "testA"
            for n, (real_A, yt) in enumerate(loader):
                real_A = real_A.to(self.device)
                with torch.no_grad():
                    out = self.seg(real_A)
                    out = out.argmax(dim=1)
                    # print(prefix, n, out.shape)
                    num_out = len(out)
                    for j in range(num_out):
                        real_idx = n*self.batch_size + j
                        if do_train_B:
                            img_name = self.trainB.fnames[real_idx][0].split('/')[-1].replace('.png', '')
                        else:
                            img_name = self.realtestA.fnames[real_idx][0].split('/')[-1].replace('.png','')
                        save_path = os.path.join(out_dir, '{}_{}_pred.png'.format(prefix, img_name))
                        print(img_name, save_path)
                        cv2.imwrite(save_path, tensor2numpy_v2(out[j])*self.seg_visual_factor)

                for n, (real_B, yt) in enumerate(self.testB_loader):
                    real_B = real_B.to(self.device)
                    with torch.no_grad():
                        out = self.seg(real_B)
                        out_np = out.clone()
                        out = out.argmax(dim=1)
                        print('testB', n, out.shape, np.min(real_B), np.max(real_B))

                        num_out = len(out)
                        for j in range(num_out):
                            real_idx = n*self.batch_size + j
                            img_name = self.testB.fnames[real_idx][0].split('/')[-1].replace('.png','')
                            save_path = os.path.join(out_dir, 'testB_{}_pred.png'.format(img_name))
                            print(img_name, save_path)
                            cv2.imwrite(save_path, tensor2numpy_v2(out[j])*self.seg_visual_factor)
                            # raw img for comparison
                            save_path = os.path.join(out_dir, 'testB_{}.png'.format(img_name))
                            cv2.imwrite(save_path, tensor2numpy(denorm(real_B[j]))*255)
                            save_path_np = os.path.join(out_dir, 'testB_{}.npy'.format(img_name))
                            out_prob = tensor2numpy_v2(out_np[j])
                            np.save(save_path_np, out_prob)

        if not self.no_gan:
            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _) in enumerate(self.realtestA_loader):
                real_A = real_A.to(self.device)

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                A2B = np.concatenate((tensor2numpy(denorm(real_A[0])),
                                      cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_A2A[0])),
                                      cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_A2B[0])),
                                      cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_A2B2A[0]))), 0)

            for n, (real_B, yt) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)

                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((tensor2numpy(denorm(real_B[0])),
                                      cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_B2B[0])),
                                      cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_B2A[0])),
                                      cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                      tensor2numpy(denorm(fake_B2A2B[0]))), 0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
        return