import torch


class OptGAN:

    def __init__(self, netG, optimizerG, netD, optimizerD, latent_n,
                 criterion, device):
        self.netG = netG
        self.optimizerG = optimizerG
        self.netD = netD
        self.optimizerD = optimizerD
        self.latent_n = latent_n
        self.criterion = criterion
        self.device = device

        self.fake_label = 0
        self.real_label = 1
        self.D_G_z1 = 0.
        self.D_G_z2 = 0.
        self.D_x = 0.
        self.errD = 0.
        self.errG = 0.

    def update_discriminator(self, data):
        self.netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), self.real_label, device=self.device)
        # Forward pass real batch through D
        output = self.netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        self.D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.latent_n, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        self.D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        self.errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()
        return label, fake

    def update_generator(self, label, fake):
        self.netG.zero_grad()
        label.fill_(self.real_label)
        output = self.netD(fake).view(-1)
        self.errG = self.criterion(output, label)
        self.errG.backward()
        self.D_G_z2 = output.mean().item()
        self.optimizerG.step()
