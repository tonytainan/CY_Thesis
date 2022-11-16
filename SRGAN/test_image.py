import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from torch.utils.data import DataLoader

from data_utils import TestDatasetFromFolder

from model import Generator
from tqdm import tqdm
import Measure # add by tony for psnr,ssim
import numpy as np # add by tony for psnr,ssim

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_8_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--gt_image_name', type=str, help='GT/HR image name')# add by tony for psnr,ssim

if __name__ == '__main__':

    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    TEST_MODE = True if opt.test_mode == 'GPU' else False
    IMAGE_NAME = opt.image_name
    MODEL_NAME = opt.model_name
    GT_IMAGE_NAME = opt.gt_image_name

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

    test_set = TestDatasetFromFolder('data', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    #image = image_ori = Image.open(IMAGE_NAME)
    #gt_image = Image.open(GT_IMAGE_NAME) # add by tony for psnr, ssim
    #image_ori = ToTensor()(image_ori)
    #gt_image = np.array(gt_image)# add by tony for psnr, ssim

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    #>>>>>>>>>>
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        lr_image = Variable(lr_image, volatile=True)
        hr_image = Variable(hr_image, volatile=True)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)
        #<<<<<<<<<<<



        #image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
        #if TEST_MODE:
        #    image = image.cuda()

        #start = time.clock()
        #out = model(image)
        #elapsed = (time.clock() - start)
        #print('cost' + str(elapsed) + 's')
        #out_img = ToPILImage()(out[0].data.cpu())


        out_img = ToPILImage()(sr_image[0].data.cpu())
        gt_image = ToPILImage()(hr_image[0].data.cpu())

        from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

        #def display_transform():
        #   return Compose([
        #     ToPILImage(),
        #     Resize((360,360), interpolation=Image.BICUBIC), # remove by tony
        #   ])


        #img_lr_8x = display_transform()(image_ori)
        #lr_image = ToTensor()(lr_image)
        #img_lr_8x = display_transform()(lr_image)


        out_img.save('out_tony/out_srf_' + str(UPSCALE_FACTOR) + '_' + image_name)

        #img_lr_8x.save('out_tony/out_lr_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)

        measure = Measure.Measure()
        out_img = np.array(out_img)    # add by tony for psnr, ssim
        gt_image = np.array(gt_image)  # add by tony for psnr, ssim
     #   print(type(out_img))
     #   print(type(hr_image))

        psnr, ssim, lpips = measure.measure(out_img, gt_image)
        #print('\r\nPSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\r\n'.format(psnr, ssim, lpips))
        print('\r\nPSNR: {:0.2f}, SSIM: {:0.2f}, LPIPS: {:0.2f}\r\n'.format(psnr, ssim, lpips)) # modify by tony 20221029
        total_psnr = total_psnr+psnr
        total_ssim = total_ssim+ssim
        total_lpips = total_lpips+lpips

    avg_psnr=total_psnr/100
    avg_ssim=total_ssim/100
    avg_lpips=total_lpips/100
    #print('\r\nAvg. PSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\r\n'.format(avg_psnr, avg_ssim, avg_lpips))
    print('\r\nAvg. PSNR: {:0.2f}, SSIM: {:0.2f}, LPIPS: {:0.2f}\r\n'.format(avg_psnr, avg_ssim, avg_lpips)) # modify by tony 20221029


