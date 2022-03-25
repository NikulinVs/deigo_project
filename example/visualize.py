import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from network.network import PVRNN

target = np.load("../dataset/motor_train.npy")
target_torch = torch.tensor(target, dtype=torch.float32)
loss = nn.MSELoss(reduction="sum")
minibatch_size = 10


def load_model(model_path):
    d_size = [10, 20, 30]
    z_size = [4, 6, 8]
    tau = [8, 4, 2]
    delay = [50, 25, 10]
    w = [0.0025, 0.005, 0.01]
    beta = [1.0, 1.0, 1.0]
    (n_seq, seq_len, output_size) = target.shape
    minibatch_size = 10
    n_minibatch = int(n_seq / minibatch_size)

    model = PVRNN(d_size, z_size, tau, w, beta, output_size, minibatch_size, n_minibatch, seq_len, delay, target_torch,
                  use_proj=True, act_func="tanh")
    state = torch.load(model_path)
    model.load_state_dict(state)

    return model


def rec_loss(model, precise=True):
    total_rec = 0
    num_batches = target.shape[0]//minibatch_size
    for i in range(num_batches):
        x, kl, wkl = model.posterior_forward(i, precise=precise)
        rec = loss(x, target_torch[i * minibatch_size: (i + 1) * minibatch_size]).item()
        total_rec += rec
    total_rec /= num_batches
    return total_rec

"""
model_random = load_model("../test_no_proj/para/epo_10000")
model_random_natural = load_model("../test_random/para/epo_10000")
model_proj = load_model("../test_proj_slow/para/epo_10000")
model_proj_natural = load_model("../test_proj/para/epo_10000")

rec_random = rec_loss(model_random)
rec_random_natural = rec_loss(model_random_natural)
rec_proj = rec_loss(model_proj)
rec_proj_natural = rec_loss(model_proj_natural)

print(f"rec loss random = {rec_random}")
print(f"rec loss random natural = {rec_random_natural}")
print(f"rec loss projection = {rec_proj}")
print(f"rec loss projection natural = {rec_proj_natural}")
"""

#total_loss_random = np.loadtxt("../test_no_proj/loss/losses_total.txt")
#total_loss_random_natural = np.loadtxt("../test_random/loss/losses_total.txt")
#total_loss_proj_natural = np.loadtxt("../test_proj/loss/losses_total.txt")
#total_loss_proj = np.loadtxt("../test_proj_slow/loss/losses_total.txt")

total_loss_random_natural = np.loadtxt("../test_random_small_z/loss/losses_total.txt")
total_loss_proj_natural = np.loadtxt("../test_proj_small_z/loss/losses_total.txt")
total_loss_random = np.loadtxt("../test_random_no_n_small_z/loss/losses_total.txt")
total_loss_proj = np.loadtxt("../test_proj_no_n_small_z/loss/losses_total.txt")

plt.rc('font', size=24)
h1, = plt.semilogy(total_loss_random)
h2, = plt.semilogy(total_loss_random_natural)
h3, = plt.semilogy(total_loss_proj)
h4, = plt.semilogy(total_loss_proj_natural)
plt.legend([h1, h2, h3, h4], ["Random posterior $\mu$", "Random natural posterior $\mu$", "Projected posterior $\mu$", "Projected natural posterior $\mu$"])
plt.show()

posterior_sigma_q_proj = torch.load("../test_proj_no_n_small_z/sequences/pos_sigmaq_10000")
posterior_sigma_q_proj = [p.detach().numpy() for p in posterior_sigma_q_proj]

posterior_sigma_p_proj = torch.load("../test_proj_no_n_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_proj = [p.detach().numpy() for p in posterior_sigma_p_proj]

posterior_sigma_p_proj_natural = torch.load("../test_proj_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_proj_natural = [p.detach().numpy() for p in posterior_sigma_p_proj_natural]

posterior_sigma_p_zero = torch.load("../test_zero_no_n_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_zero = [p.detach().numpy() for p in posterior_sigma_p_zero]

posterior_sigma_p_zero_natural = torch.load("../test_zero_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_zero_natural = [p.detach().numpy() for p in posterior_sigma_p_zero_natural]

posterior_sigma_q_proj_natural = torch.load("../test_proj_small_z/sequences/pos_sigmaq_10000")
posterior_sigma_q_proj_natural = [q.detach().numpy() for q in posterior_sigma_q_proj_natural]

posterior_sigma_q_random = torch.load("../test_random_no_n_small_z/sequences/pos_sigmaq_10000")
posterior_sigma_q_random = [p.detach().numpy() for p in posterior_sigma_q_random]

posterior_sigma_p_random = torch.load("../test_random_no_n_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_random = [p.detach().numpy() for p in posterior_sigma_p_random]

posterior_sigma_p_random_natural = torch.load("../test_random_small_z/sequences/pos_sigmap_10000")
posterior_sigma_p_random_natural = [p.detach().numpy() for p in posterior_sigma_p_random_natural]

layer = 0

for i in range(10):
    h3, = plt.plot(posterior_sigma_p_proj[layer][i,:,:].mean(axis=1), color='r')
    h4, = plt.plot(posterior_sigma_p_proj_natural[layer][i,:,:].mean(axis=1), color='g')
    h1, = plt.plot(posterior_sigma_p_random[layer][i,:,:].mean(axis=1), color='b')
    h2, = plt.plot(posterior_sigma_p_random_natural[layer][i,:,:].mean(axis=1), color='y')
    #h5, = plt.plot(posterior_sigma_p_zero[layer][i,:,:].mean(axis=1), color='violet')
    #h6, = plt.plot(posterior_sigma_p_zero_natural[layer][i, :, :].mean(axis=1), color='lightblue')
plt.legend([h1, h2, h3, h4], ["Random prior $\sigma$", "Random natural prior $\sigma$", "Projected prior $\sigma$", "Projected natural prior $\sigma$"])
plt.ylabel("Prior Sigma")
plt.xlabel("Timestep")
plt.show()

#exit(0)

target = np.load("../dataset/motor_train.npy")
x_traj = torch.load("../test_proj_small_z/sequences/pos_x_10000")
x_traj = x_traj.detach().numpy()

plt.plot(target[0,:,7])
plt.plot(x_traj[0,:,7], color='r')
plt.show()

delay = 50
tau = 8.
z_size = 4
seq_len = target.shape[1]
num_samples = target.shape[0]
random_projection = np.random.randn(delay, target.shape[2], z_size)
projected_traj = np.zeros((num_samples, seq_len, z_size))

for k in range(seq_len):
    l_idx = min(k, seq_len - delay)
    pz = np.einsum("djz, bdj -> bz", random_projection,
                      target[:, l_idx:l_idx + delay])
    pz = pz - pz.mean(axis=0)
    pz = pz / pz.std(axis=0)

    if k == 0:
        projected_traj[:, k] = pz
    else:
        projected_traj[:, k] = (1 - 1. / tau) * projected_traj[:, k - 1] + pz / tau

plt.plot(projected_traj[0, :, 1])

projected_traj = np.zeros((num_samples, seq_len, z_size))

for k in range(seq_len):
    l_idx = min(k, seq_len - delay)
    pz = np.einsum("djz, bdj -> bz", random_projection,
                   target[:, l_idx:l_idx + delay])
    pz = pz - pz.mean(axis=0)
    pz = pz / pz.std(axis=0)

    projected_traj[:, k] = pz

plt.plot(projected_traj[0, :, 1])
plt.show()


