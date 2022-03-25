import torch
import torch.nn as nn


# class for one pvrnn layer
class PVRNNLayer(nn.Module):
    def __init__(self, d_size: int, z_size: int, tau: torch.FloatTensor, w: float, beta: float, minibatch_size: int, n_minibatch: int, seq_len: int, delay: int, target: torch.Tensor, use_proj=False, use_natural=True, input_size=0, device="cpu"):
        super(PVRNNLayer, self).__init__()
        # hyper parameters
        self.d_size = d_size # dim of d
        self.z_size = z_size # dim of z
        self.tau = tau       # time constant for MTRNN
        self.w = w           # meta-prior
        self.beta = beta     # beta (w at t=1)
        self.input_size = input_size # d dim of higher layer if not top layer; otherwise 0
        self.minibatch_size = minibatch_size # #sequences in one minibatch
        self.n_minibatch = n_minibatch       # #minibatches in dataset
        self.seq_len = seq_len               # sequence length (all sequence have to be the same length)
        self.device = device                 # "cpu" or "cuda"
        self.use_natural = use_natural
        # weights
        self.d_to_h = nn.Linear(d_size, d_size)

        self.z_to_h = nn.Linear(z_size, d_size)
        self.d_to_mu = nn.Linear(d_size, z_size)
        self.d_to_sigma = nn.Linear(d_size, z_size)
        if input_size != 0:
            self.hd_to_h = nn.Linear(input_size, d_size)
            self.up_connection = True
        else:
            self.up_connection = False
        self.random_projection = torch.randn((delay, target.shape[2], z_size), device=device)
        projected_traj = [torch.zeros((minibatch_size, seq_len, z_size), requires_grad=False, device=device) for _ in range(n_minibatch)]
        for i in range(n_minibatch):
            for k in range(seq_len):
                l_idx = min(k, seq_len-delay)
                pz = torch.einsum("djz, bdj -> bz", self.random_projection, target[i*minibatch_size:(i+1)*minibatch_size, l_idx:l_idx + delay])
                pz = pz - pz.mean(axis=0)
                pz = pz / pz.std(axis=0)
                if k == 0:
                    projected_traj[i][:, k] = pz
                else:
                    projected_traj[i][:, k] = (1. - 1. / self.tau) * projected_traj[i][:, k-1] + pz / self.tau
        # A
        if use_proj:
            self.Amu = nn.ParameterList(nn.Parameter(projected_traj[i].clone().detach().requires_grad_(True), requires_grad=True) for i in range(n_minibatch))
        else:
            self.Amu = nn.ParameterList(nn.Parameter(torch.randn(minibatch_size, seq_len, z_size, device=device)) for _ in range(n_minibatch))
        self.Asigma = nn.ParameterList(nn.Parameter(torch.randn(minibatch_size, seq_len, z_size, device=device)) for _ in range(n_minibatch))
        # initial state
        self.d0 = torch.zeros(minibatch_size, d_size, device=device)
        self.h0 = torch.zeros(minibatch_size, d_size, device=device)

    def freeze_weights(self, freeze=True):
        self.z_to_h.weight.data.requires_grad = not freeze
        self.z_to_h.bias.data.requires_grad = not freeze
        self.d_to_h.weight.data.requires_grad = not freeze
        self.d_to_h.bias.data.requires_grad = not freeze
        self.d_to_mu.weight.data.requires_grad = not freeze
        self.d_to_mu.bias.data.requires_grad = not freeze
        self.d_to_sigma.weight.data.requires_grad = not freeze
        self.d_to_sigma.bias.data.requires_grad = not freeze

        self.z_to_h.weight.requires_grad = not freeze
        self.z_to_h.bias.requires_grad = not freeze
        self.d_to_h.weight.requires_grad = not freeze
        self.d_to_h.bias.requires_grad = not freeze
        self.d_to_mu.weight.requires_grad = not freeze
        self.d_to_mu.bias.requires_grad = not freeze
        self.d_to_sigma.weight.requires_grad = not freeze
        self.d_to_sigma.bias.requires_grad = not freeze
        if self.up_connection:
            self.hd_to_h.weight.requires_grad = not freeze
            self.hd_to_h.bias.requires_grad = not freeze

            self.hd_to_h.weight.data.requires_grad = not freeze
            self.hd_to_h.bias.data.requires_grad = not freeze

    def initialize(self, minibatch_ind):
        self.d = self.d0
        self.h = self.h0
        self.t = 0
        self.kl = 0.
        self.wkl = 0.
        self.minibatch_ind = minibatch_ind

    def compute_mu_sigma(self):
        if self.t < self.seq_len:
            self.muq    = self.Amu[self.minibatch_ind][:, self.t, :].tanh()
            self.sigmaq = self.Asigma[self.minibatch_ind][:, self.t, :].exp()

            if self.use_natural:
                sigmaq = self.sigmaq.detach().clone()
                self.muq.register_hook(lambda grad: grad * sigmaq * sigmaq)
                self.sigmaq.register_hook(lambda grad: grad * sigmaq * sigmaq / 2.)

            if self.t == 0:
                self.mup = torch.zeros(self.minibatch_size, self.z_size)
                self.sigmap = torch.ones(self.minibatch_size, self.z_size)
            else:
                self.mup    = self.d_to_mu(self.d).tanh()
                self.sigmap = self.d_to_sigma(self.d).exp()
        else:
            self.mup = self.d_to_mu(self.d).tanh()
            self.sigmap = self.d_to_sigma(self.d).exp()

        if self.mup.requires_grad and self.use_natural:
            sigmap = self.sigmap.detach().clone()
            self.mup.register_hook(lambda grad: grad * sigmap * sigmap)
            self.sigmap.register_hook(lambda grad: grad * sigmap * sigmap / 2.)

    def sample_zq(self, precise=False):
        if precise:
            self.z = self.muq
        else:
            self.z = self.muq + torch.randn(self.sigmaq.shape) * self.sigmaq

    def sample_zp(self):
        self.z = self.mup + torch.randn(self.sigmap.shape) * self.sigmap

    def compute_mtrnn(self, hd=None):
        if hd is None:
            h = (1. - 1. / self.tau) * self.h + (self.d_to_h(self.d) + self.z_to_h(self.z)) / self.tau
        else:
            h = (1. - 1. / self.tau) * self.h + (self.d_to_h(self.d) + self.z_to_h(self.z) + self.hd_to_h(hd)) / self.tau
        d = h.tanh()
        self.h = h
        self.d = d

    def compute_kl(self):
        return torch.sum( ((self.sigmap / self.sigmaq) + 1e-20).log() + 0.5 * ((self.mup - self.muq).pow(2) + self.sigmaq.pow(2))
                         / self.sigmap.pow(2) + 0.5 ) / self.z_size

    def posterior_step(self, hd=None, precise=False):
        self.compute_mu_sigma()
        self.sample_zq(precise=precise)
        self.compute_kl()
        self.compute_mtrnn(hd)
        _kl = self.compute_kl()
        self.kl += _kl
        if self.t == 0:
            self.wkl += self.beta * _kl
        else:
            self.wkl += self.w * _kl
        self.t += 1

    def prior_step(self, hd=None):
        self.compute_mu_sigma()
        self.sample_zp()
        self.compute_mtrnn(hd)
        self.t += 1
