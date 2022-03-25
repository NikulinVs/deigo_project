import torch
import torch.nn as nn
from network.layer import PVRNNLayer
from network.output import Output

# network class using PVRNNLayer and Output
class PVRNN(nn.Module):
    def __init__(self, d_size: list, z_size: list, tau: list, w: list, beta: list, output_size: int, minibatch_size: int, n_minibatch: int, seq_len, delay: list, target: torch.Tensor, use_proj=False, use_natural=True, act_func="sigmoid", device="cpu"):
        super(PVRNN, self).__init__()
        self.d_size         = d_size
        self.z_size         = z_size
        self.tau            = tau
        self.w              = w
        self.beta           = beta
        self.output_size    = output_size
        self.minibatch_size = minibatch_size
        self.n_minibatch    = n_minibatch
        self.seq_len        = seq_len
        self.device         = device
        self.n_layer        = len(d_size)
        self.delay          = delay
        self.target         = target
        self.use_proj       = use_proj

        self.layers = nn.ModuleList()

        # instantiate layers from top to bottom
        self.layers.append(PVRNNLayer(d_size[0], z_size[0], tau[0], w[0], beta[0], minibatch_size, n_minibatch, seq_len, delay[0], target, use_proj=use_proj, use_natural=use_natural))
        for l in range(1, self.n_layer):
            self.layers.append(PVRNNLayer(d_size[l], z_size[l], tau[l], w[l], beta[l], minibatch_size, n_minibatch, seq_len, delay[l], target, use_proj=use_proj, use_natural=use_natural, input_size=d_size[l-1]))
        self.output = Output(d_size[-1], output_size, act_func)

    def initialize(self, minibatch_ind):
        for layer in self.layers:
            layer.initialize(minibatch_ind)

    def freeze_layers(self, freeze=True):
        for layer in self.layers:
            layer.freeze_weights(freeze)

    def posterior_step(self, precise=False):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.posterior_step(precise=precise)
            else:
                layer.posterior_step(self.layers[i - 1].d, precise=precise)
        return self.output(self.layers[-1].d)

    def prior_step(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.prior_step()
            else:
                layer.prior_step(self.layers[i -1].d)
        return self.output(self.layers[-1].d)

    def posterior_forward(self, minibatch_ind: int, precise=False):
        self.initialize(minibatch_ind)
        x = torch.zeros(self.minibatch_size, self.seq_len, self.output_size, device="cpu")
        kl, wkl = 0., 0.
        for t in range(self.seq_len):
            x[:, t, :] = self.posterior_step()
        for layer in self.layers:
            kl += layer.kl
            wkl += layer.wkl
        return x, kl, wkl

    def posterior_record(self, minibatch_ind: int):
        self.initialize(minibatch_ind)
        x = torch.zeros(self.minibatch_size, self.seq_len, self.output_size, device="cpu")
        d, mup, sigmap, muq, sigmaq = [], [], [], [], []
        for _d, _z in zip(self.d_size, self.z_size):
            d.append(torch.zeros(self.minibatch_size, self.seq_len, _d, device="cpu"))
            mup.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))
            sigmap.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))
            muq.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))
            sigmaq.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))

        for t in range(self.seq_len):
            x[:, t, :] = self.posterior_step()
            for i, layer in enumerate(self.layers):
                d[i][:, t, :] = layer.d
                mup[i][:, t, :] = layer.mup
                sigmap[i][:, t, :] = layer.sigmap
                muq[i][:, t, :] = layer.muq
                sigmaq[i][:, t, :] = layer.sigmaq
        return x, d, mup, sigmap, muq, sigmaq

    def prior_forward(self, minibatch_ind: int):
        self.initialize(minibatch_ind)
        x = torch.zeros(self.minibatch_size, self.seq_len, self.output_size, device="cpu")
        d, mu, sigma = [], [], []
        for _d, _z in zip(self.d_size, self.z_size):
            d.append(torch.zeros(self.minibatch_size, self.seq_len, _d, device="cpu"))
            mu.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))
            sigma.append(torch.zeros(self.minibatch_size, self.seq_len, _z, device="cpu"))
        x[:, 0, :] = self.posterior_step()
        for i, layer in enumerate(self.layers):
            d[i][:, 0, :] = layer.d
            mu[i][:, 0, :] = layer.muq
            sigma[i][:, 0, :] = layer.sigmaq

        for t in range(1, self.seq_len):
            x[:, t, :] = self.prior_step()
            for i, layer in enumerate(self.layers):
                d[i][:, t, :] = layer.d
                mu[i][:, t, :] = layer.mup
                sigma[i][:, t, :] = layer.sigmap
        return x, d, mu, sigma

    def save_param(self, fn):
        param = self.state_dict()
        torch.save(param, fn)

    def load_param(self, fn):
        param = torch.load(fn)
        self.load_state_dict(param)


