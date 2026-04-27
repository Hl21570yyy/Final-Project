import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import inter_contrastive_Loss, intra_contrastive_loss

class Autoencoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        return self._encoder(x)

    def decoder(self, latent):
        return self._decoder(latent)

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    def __init__(self, prediction_dim, activation='relu', batchnorm=True):
        super(Prediction, self).__init__()
        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 0:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class ReCP():
    def __init__(self, config):
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        self.autoencoder_a = Autoencoder(config['Autoencoder']['arch1'],
                                         config['Autoencoder']['activations1'],
                                         config['Autoencoder']['batchnorm'])
        self.autoencoder_s = Autoencoder(config['Autoencoder']['arch2'],
                                         config['Autoencoder']['activations2'],
                                         config['Autoencoder']['batchnorm'])
        self.autoencoder_d = Autoencoder(config['Autoencoder']['arch2'],
                                         config['Autoencoder']['activations2'],
                                         config['Autoencoder']['batchnorm'])

        self.max_net = torch.nn.Sequential(nn.Softmax(dim=1))
        self.a2mo = Prediction(self._dims_view1)
        self.mo2a = Prediction(self._dims_view2)

    def to_device(self, device):
        self.autoencoder_a.to(device)
        self.autoencoder_s.to(device)
        self.autoencoder_d.to(device)
        self.a2mo.to(device)
        self.mo2a.to(device)

    def train(self, config, redata, xs, optimizer, scheduler, device):
        for epoch in range(config['training']['epoch']):
            xs_aug_raw = redata.get_aug(seed=epoch)
            xs_aug = []
            for view in range(len(xs)):
                li = []
                for i in range(len(xs_aug_raw[view])):
                    li.append(torch.from_numpy(xs_aug_raw[view][i]).float().to(device))
                xs_aug.append(li)

            z_1 = self.autoencoder_a.encoder(xs[0])
            z_2_1 = self.autoencoder_s.encoder(xs[1])
            z_2_2 = self.autoencoder_d.encoder(xs[2])
            z_2 = (z_2_1 + z_2_2) / 2

            z_1_aug_list, z_2_aug_list = [], []
            for aug in range(len(xs_aug[0])):
                z_1_aug_list.append(self.autoencoder_a.encoder(xs_aug[0][aug]))

            for aug in range(len(xs_aug[1])):
                z_2_1_aug = self.autoencoder_s.encoder(xs_aug[1][aug])
                z_2_2_aug = self.autoencoder_d.encoder(xs_aug[2][aug])
                z_2_aug_list.append((z_2_1_aug + z_2_2_aug) / 2)

            recon1 = F.mse_loss(self.autoencoder_a.decoder(z_1), xs[0])
            recon2 = F.mse_loss(self.autoencoder_s.decoder(z_2_1), xs[1])
            recon3 = F.mse_loss(self.autoencoder_d.decoder(z_2_2), xs[2])
            reconstruction_loss = config['training']['sigma'] * recon1 + recon2 + recon3

            intra_loss_1 = intra_contrastive_loss(z_1, z_1_aug_list)
            intra_loss_2 = intra_contrastive_loss(z_2, z_2_aug_list)
            intra_loss = config['training']['sigma'] * intra_loss_1 + intra_loss_2

            z_1_cl = self.max_net(z_1)
            z_2_cl = self.max_net(z_2)
            cl_loss = inter_contrastive_Loss(z_1_cl, z_2_cl, config['training']['alpha'])

            p2mo, _ = self.a2mo(z_1)
            mo2p, _ = self.mo2a(z_2)
            pre1 = F.mse_loss(p2mo, z_2)
            pre2 = F.mse_loss(mo2p, z_1)
            dualprediction_loss = config['training']['sigma'] * pre1 + pre2

            loss = (cl_loss
                    + reconstruction_loss * config['training']['lambda2']
                    + intra_loss * config['training']['lambda1'])

            if epoch >= config['training']['start_dual_prediction']:
                loss += dualprediction_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            if (epoch + 1) % config['print_num'] == 0:
                print(f"Epoch {epoch+1:>4d}/{config['training']['epoch']} | "
                      f"Loss={loss.item():.4f} | "
                      f"recon={reconstruction_loss.item():.4f} | "
                      f"intra={intra_loss.item():.4f} | "
                      f"inter={cl_loss.item():.4f} | "
                      f"dual={dualprediction_loss.item():.4f}")

    @torch.no_grad()
    def get_embeddings(self, xs):
        """
        提取单视图区域嵌入，以字典形式返回，方便下游任务消融使用。
        """
        self.autoencoder_a.eval()
        self.autoencoder_s.eval()
        self.autoencoder_d.eval()

        latent_a = self.autoencoder_a.encoder(xs[0])    # (R, 48)
        latent_s = self.autoencoder_s.encoder(xs[1])    # (R, 48)
        latent_d = self.autoencoder_d.encoder(xs[2])    # (R, 48)
        latent_m = (latent_s + latent_d) / 2            # (R, 48)

        self.autoencoder_a.train()
        self.autoencoder_s.train()
        self.autoencoder_d.train()

        return {
            'latent_a': latent_a.cpu().numpy(),
            'latent_m': latent_m.cpu().numpy()
        }