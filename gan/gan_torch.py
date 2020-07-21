import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models.nets_torch import weights_init, Discriminator, Generator
from models.opt_torch import OptGAN
from utils.gan_funcs import plot_real_vs_fake_images

# print_every = 50
print_every = 1
ngpu = 1
epochs = 2
batch_size = 256
seed = 9331
random.seed(seed)
torch.manual_seed(seed)
dataroot = "data/celeba"
workers = 2
lr = 0.0002
beta1 = 0.5
image_size = 64
num_channels = 3
latent_n = 100
ngf = 64
ndf = 64

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu, latent_n, ngf, num_channels).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

netD = Discriminator(ngpu, num_channels, ndf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, latent_n, 1, 1, device=device)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
img_list, G_losses, D_losses = [], [], []
iters = 0

opt_gan = OptGAN(netG, optimizerG, netD, optimizerD, latent_n,
                 criterion, device)

print("Starting Training Loop...")
for epoch in range(epochs):
    t0 = time.time()
    for i, data in enumerate(dataloader, 0):
        tic = time.time()
        label, fake = opt_gan.update_discriminator(data)
        opt_gan.update_generator(label, fake)
        toc = time.time()

        if i % print_every == 0:
            message = f'[{epoch:d}/{epochs:d}][{i:d}/{len(dataloader):d}] '
            message += f'|| Loss_D: {opt_gan.errD.item():2.2f} '
            message += f'|| Loss_G: {opt_gan.errG.item():2.2f} '
            message += f'|| D(x): {opt_gan.D_x:2.2f} '
            message += f'|| D(G(z)): {opt_gan.D_G_z1:2.2f} / {opt_gan.D_G_z2:2.2f} '
            message += f'|| {toc - tic:2.2f} sec'
            print(message)

        G_losses.append(opt_gan.errG.item())
        D_losses.append(opt_gan.errD.item())

        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1
    t1 = time.time()
    print(f'Epoch took {t1 - t0:2.2f} sec')

plot_real_vs_fake_images(dataloader, img_list, device)
