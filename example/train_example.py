import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..")
from network.network import PVRNN
from pathlib import Path


def train_cross_validation(use_proj, use_natural, dir) -> None:
    target = torch.from_numpy(np.load("../dataset/motor_train.npy")).type(torch.FloatTensor)

    # example of parameter settings
    d_size = [10, 20, 30]
    z_size = [1, 2, 3]
    tau = [8, 4, 2]
    delay = [50, 25, 10]
    w = [0.0025, 0.005, 0.01]
    beta = [1.0, 1.0, 1.0]
    (n_seq, seq_len, output_size) = target.shape
    minibatch_size = 5
    n_minibatch = int(n_seq / minibatch_size)
    n_epoch = 7000
    n_epoch_val = 4000
    assert (target.shape[0] % minibatch_size) == 0

    save_dir = Path(__file__).parent / ".." / dir
    save_dir.mkdir(exist_ok=True)

    seq_dir = save_dir / "sequences"
    para_dir = save_dir / "para"
    loss_dir = save_dir / "loss"

    seq_dir.mkdir(exist_ok=True)
    para_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(exist_ok=True)

    for batch_out in range(1, n_minibatch, 2):
        model = PVRNN(d_size, z_size, tau, w, beta, output_size, minibatch_size, n_minibatch, seq_len, delay, target,
                      use_proj=use_proj, use_natural=use_natural, act_func="tanh", )
        loss = nn.MSELoss(reduction="sum")

        opt = optim.Adam(model.parameters(), lr=1e-3)
        print(f"Start training with batch #{batch_out} out...")
        for epo in range(n_epoch):
           rec_epoch = 0
           kl_epoch = 0
           total_epoch = 0
           #for i in range(n_minibatch):
           #    if i == batch_out:
           #        continue
           x, kl, wkl = model.posterior_forward(batch_out)
           _rec = loss(x, target[batch_out * minibatch_size: (batch_out + 1) * minibatch_size])
           _loss = _rec + wkl
           opt.zero_grad()
           _loss.backward()
           opt.step()

           rec_epoch += _rec.detach().item()
           kl_epoch += kl.detach().item()
           total_epoch += _loss.detach().item()

           print("Training. epo: {} rec: {:.2f} kl: {:.2f}".format(epo, rec_epoch, kl_epoch))

        model.freeze_layers()

        f_loss_rec = open(f'{loss_dir}/losses_{batch_out}_rec.txt', 'w')
        f_loss_kl = open(f'{loss_dir}/losses_{batch_out}_kl.txt', 'w')
        f_loss_total = open(f'{loss_dir}/losses_{batch_out}_total.txt', 'w')

        for epo in range(n_epoch_val):
            rec_epoch = 0
            kl_epoch = 0
            total_epoch = 0
            for i in range(n_minibatch):
                if i == batch_out:
                    continue
                x, kl, wkl = model.posterior_forward(i)
                _rec = loss(x, target[i * minibatch_size: (i + 1) * minibatch_size])
                _loss = _rec + wkl
                opt.zero_grad()
                _loss.backward()
                opt.step()
                rec_epoch += _rec.detach().item()
                kl_epoch += kl.detach().item()
                total_epoch += _loss.detach().item()

            print("Validation. epo: {} rec: {:.2f} kl: {:.2f}".format(epo, rec_epoch, kl_epoch))

            if (epo + 1) % 1000 == 0:
               x, d, mup, sigmap, muq, sigmaq = model.posterior_record(0)
               torch.save(x, seq_dir / "pos_x_{}_{}".format(batch_out, epo + 1))
               torch.save(d, seq_dir / "pos_d_{}_{}".format(batch_out, epo + 1))
               torch.save(mup, seq_dir / "pos_mup_{}_{}".format(batch_out, epo + 1))
               torch.save(sigmap, seq_dir / "pos_sigmap_{}_{}".format(batch_out, epo + 1))
               torch.save(muq, seq_dir / "pos_muq_{}_{}".format(batch_out, epo + 1))
               torch.save(sigmaq, seq_dir / "pos_sigmaq_{}_{}".format(batch_out, epo + 1))

               x, d, mu, sigma = model.prior_forward(batch_out)
               torch.save(x, seq_dir / "pri_x_{}_{}".format(batch_out, epo + 1))
               torch.save(d, seq_dir / "pri_d_{}_{}".format(batch_out, epo + 1))
               torch.save(mu, seq_dir / "pri_mu_{}_{}".format(batch_out, epo + 1))
               torch.save(sigma, seq_dir / "pri_sigma_{}_{}".format(batch_out, epo + 1))

               model.save_param(para_dir / "epo_{}_{}".format(batch_out, epo + 1))

            f_loss_rec.write(f'{rec_epoch}\n')
            f_loss_kl.write(f'{kl_epoch}\n')
            f_loss_total.write(f'{total_epoch}\n')

        f_loss_kl.close()
        f_loss_rec.close()
        f_loss_total.close()


def train(use_proj, use_natural, dir) -> None:
    target = torch.from_numpy(np.load("../dataset/motor_train.npy")).type(torch.FloatTensor)

    # example of parameter settings
    d_size = [10, 20, 30]
    z_size = [1, 2, 3]
    tau    = [8, 4, 2]
    delay  = [50, 25, 10]
    w      = [0.0025, 0.005, 0.01]
    beta   = [1.0, 1.0, 1.0]
    (n_seq, seq_len, output_size) = target.shape
    minibatch_size = 10
    n_minibatch = int(n_seq / minibatch_size)
    n_epoch = 10000
    assert (target.shape[0]%minibatch_size) == 0

    save_dir = Path(__file__).parent / ".." / dir
    save_dir.mkdir(exist_ok=True)

    seq_dir  = save_dir / "sequences"
    para_dir = save_dir / "para"
    loss_dir = save_dir / "loss"

    seq_dir.mkdir(exist_ok=True)
    para_dir.mkdir(exist_ok=True)
    loss_dir.mkdir(exist_ok=True)

    model = PVRNN(d_size, z_size, tau, w, beta, output_size, minibatch_size, n_minibatch, seq_len, delay, target, use_proj=use_proj, use_natural=use_natural, act_func="tanh",)
    loss  = nn.MSELoss(reduction="sum")
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    rec_loss, kl_loss, total_loss = [], [], []

    f_loss_rec = open(f'{loss_dir}/losses_rec.txt', 'w')
    f_loss_kl = open(f'{loss_dir}/losses_kl.txt', 'w')
    f_loss_total = open(f'{loss_dir}/losses_total.txt', 'w')

    print("Start training...")
    for epo in range(n_epoch):
        rec_epoch = 0
        kl_epoch = 0
        total_epoch = 0
        for i in range(n_minibatch):
            x, kl, wkl = model.posterior_forward(i)
            _rec = loss(x, target[i * minibatch_size: (i+1) * minibatch_size])
            _loss = _rec + wkl
            opt.zero_grad()
            _loss.backward()
            opt.step()

            rec_epoch += _rec.detach().item()
            kl_epoch  += kl.detach().item()
            total_epoch += _loss.detach().item()
        
        rec_loss.append(rec_epoch)
        kl_loss.append(kl_epoch)
        total_loss.append(total_epoch)
        print("epo: {} rec: {:.2f} kl: {:.2f}".format(epo, rec_epoch, kl_epoch))

        f_loss_rec.write(f'{rec_epoch}\n')
        f_loss_kl.write(f'{kl_epoch}\n')
        f_loss_total.write(f'{total_epoch}\n')

        # saving sequences and parameters every 100 epochs
        if (epo + 1) % 1000 == 0:
            x, d, mup, sigmap, muq, sigmaq = model.posterior_record(0)
            torch.save(x, seq_dir / "pos_x_{}".format(epo+1))
            torch.save(d, seq_dir / "pos_d_{}".format(epo+1))
            torch.save(mup, seq_dir / "pos_mup_{}".format(epo+1))
            torch.save(sigmap, seq_dir / "pos_sigmap_{}".format(epo+1))
            torch.save(muq, seq_dir / "pos_muq_{}".format(epo+1))
            torch.save(sigmaq, seq_dir / "pos_sigmaq_{}".format(epo+1))

            x, d, mu, sigma = model.prior_forward(0)
            torch.save(x, seq_dir / "pri_x_{}".format(epo+1))
            torch.save(d, seq_dir / "pri_d_{}".format(epo+1))
            torch.save(mu, seq_dir / "pri_mu_{}".format(epo+1))
            torch.save(sigma, seq_dir / "pri_sigma_{}".format(epo+1))

            model.save_param(para_dir / "epo_{}".format(epo+1))

    f_loss_kl.close()
    f_loss_rec.close()
    f_loss_total.close()


if __name__ == "__main__":
    torch.random.manual_seed(0)
    np.random.seed(0)

    train_cross_validation(True, True, "validation/test_proj")
    train_cross_validation(False, True, "validation/test_random")

    train_cross_validation(True, False, "validation/test_proj_no_n")
    train_cross_validation(False, False, "validation/test_random_no_n")
