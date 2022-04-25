from RBM import *
from utils import *
from RBMbase import *
from RBMucd import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns   


if not os.path.exists('./plot'):
    os.makedirs('./plot')
    
# parameters 
n_hidden = 9
n_visible = 25
n_train = 10000
n_valid = 3000
seed = 521
max_epoch = 2000
k = 10
show = 10
np.random.seed(888)



for method in ['sparse', 'random']:
    for lr in [0.1, 0.01, 0.005, 0.0001, 0.0005]:
            for batchsize in [100, 500, 1000]:
                if method =='sparse':
                    plist = [0.1, 0.3, 0.5]
                    #lr = 0.05
                if method == 'random':
                    plist = [None]
                    #lr = 0.05
                if method =='real':
                    lr = 0.01    
                for p in plist:
                    #======generate true weights
                    W_true = generate_W(method, n_hidden, n_visible, p=p)
                    W_true = W_true.clone()
                    n_hidden, n_visible = W_true.shape
                    
                    # Generate initial weights
                    # warm start
                    if method=='random':    
                        W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                    0.6 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                    if method=='sparse':
                        W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                    0.6 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                    
                    if method == 'real':
                        W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                    0.1*min(torch.norm(W_true),6)/(n_hidden*n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                               

                    # Generate data from W_true
                    rbm_true =  RBM(n_visible, n_hidden, n_train + n_valid + 5000, lr, batchsize, seed=seed, W_init = torch.clone(W_true))
                    samples = rbm_true.samp(n_train + 5000)
                    samples_converged = samples[5001:,]
                    shuffle_indices = np.arange(0, n_train + n_valid)
                    np.random.shuffle(shuffle_indices)
                    samples_train = samples_converged[shuffle_indices[:n_train],]
                    samples_valid = samples_converged[shuffle_indices[n_train:],]
                    print(samples_train.shape)
                    print(samples_valid.shape)

                    # train baseline 
                    #rbm_train_base = RBM_base(n_visible, n_hidden, k, W_init.detach().numpy(), lr=lr, minibatch_size=batchsize, seed= seed)
                    rbm_train_base = RBM(n_visible, n_hidden, k, W_init=torch.clone(W_init), sparsity = p, masked = False, lr=lr/5, minibatch_size=batchsize, seed= seed)
                    training_loss_base, valid_loss_base, dist_weights_base, ratio_base, err_train_base, err_valid_base, energy_train_base, energy_valid_base = rbm_train_base.train(samples_train, samples_valid, W_true, max_epoch, show)

                    fig, ((ax1, ax2, ax8, ax3),(ax4, ax5, ax7, ax6)) = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
                    ax1.plot(np.arange(len(training_loss_base)), training_loss_base, label = 'Train')
                    ax1.plot(np.arange(len(training_loss_base)), valid_loss_base, label = 'Validation')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Baseline loss')
                    ax1.legend()

                    # ax2.plot(np.arange(len(training_loss_base)), dist_weights_base)
                    # ax2.set_xlabel('Iterations')
                    # ax2.set_ylabel('Distance between weights')
                    # ax2.set_title('Baseline weight diff')
                    ax2.plot(show*np.arange(len(err_train_base)), err_train_base, label='Train')
                    ax2.plot(show*np.arange(len(err_train_base)), err_valid_base,label='Validation')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Reconstruction Error')
                    ax2.set_title('Baseline Reconstruction')
                    ax2.legend()

                    ax8.plot(show*np.arange(len(energy_train_base)), energy_train_base, label = 'Train')
                    ax8.plot(show*np.arange(len(energy_valid_base)), energy_valid_base, label = 'Validation')
                    ax8.set_xlabel('Epoch')
                    ax8.set_ylabel('Free energy')
                    ax8.set_title('Baseline energy')
                    ax8.legend()

                    ax3.plot(np.arange(len(training_loss_base)), ratio_base)
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Ratio of weight difference')
                    ax3.set_ylim(bottom=0)
                    ax3.set_title('Baseline weight diff')


                    # train masked 
                    rbm_train_mask = RBM(n_visible, n_hidden, k, lr*10, batchsize, seed=seed, W_init=torch.clone(W_init), masked=True)
                    training_loss_mask, valid_loss_mask, dist_weights_mask, ratio_mask, err_train_mask, err_valid_mask, energy_train_mask, energy_valid_mask = rbm_train_mask.train(samples_train, samples_valid, W_true, max_epoch, show)

                    ax4.plot(np.arange(len(training_loss_mask)), training_loss_mask, label = 'Train')
                    ax4.plot(np.arange(len(valid_loss_mask)), valid_loss_mask, label = 'Validation')
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Loss')
                    ax4.set_title('Masked loss')
                    ax4.legend()

                    ax7.plot(show*np.arange(len(energy_train_mask)), energy_train_mask, label = 'Train')
                    ax7.plot(show*np.arange(len(energy_valid_mask)), energy_valid_mask, label = 'Validation')
                    ax7.set_xlabel('Epoch')
                    ax7.set_ylabel('Free energy')
                    ax7.set_title('Masked energy')
                    ax7.legend()

                    # ax5.plot(np.arange(len(training_loss_mask)), dist_weights_mask)
                    # ax5.set_xlabel('Iterations')
                    # ax5.set_ylabel('Distance between weights')
                    # ax5.set_title('Masked weight diff')
                    ax5.plot(show*np.arange(len(err_train_mask)), err_train_mask, label='Train')
                    ax5.plot(show*np.arange(len(err_valid_mask)), err_valid_mask,label='Validation')
                    ax5.set_xlabel('Epoch')
                    ax5.set_ylabel('Reconstruction Error')
                    ax5.set_title('Masked Reconstruction')
                    ax5.legend()

                    ax6.plot(np.arange(len(training_loss_base)), ratio_mask)
                    ax6.set_xlabel('Iterations')
                    ax6.set_ylabel('Ratio of weight difference')
                    ax6.set_ylim(bottom=0)
                    ax6.set_title('Masked weight diff')


                    fig.tight_layout()
                    plt.savefig("./plot/RBM_W{0}_h{1}_v{2}_lr{3}_batch{4}_p{5}_epoch{6}.png".format(method,n_hidden,n_visible,lr,batchsize,p,max_epoch), format="png")


                    # plot W
                    fig = plt.figure()
                    cnt = 0
                    for i in range(n_hidden):
                        cnt += 1
                        ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                             int(np.sqrt(n_hidden)), cnt)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        im = plt.imshow(rbm_train_base.W.detach().numpy()[i, :].reshape([5, 5]),
                                        vmin = -max(abs(rbm_train_base.W.detach().numpy()[i, :])), 
                                        vmax = max(abs(rbm_train_base.W.detach().numpy()[i, :])), 
                                        cmap='coolwarm')

                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    plt.savefig("./plot/base_W{0}_h{1}_v{2}_lr{3}_batch{4}_p{5}_epoch{6}.png".format(method,n_hidden,n_visible,lr,batchsize,p,max_epoch), format="png")

                                   # plot W
                    fig = plt.figure()
                    cnt = 0
                    for i in range(n_hidden):
                        cnt += 1
                        ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                             int(np.sqrt(n_hidden)), cnt)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        im = plt.imshow(rbm_train_mask.W.detach().numpy()[i, :].reshape([5, 5]),
                                        vmin = -max(abs(rbm_train_mask.W.detach().numpy()[i, :])), 
                                        vmax = max(abs(rbm_train_mask.W.detach().numpy()[i, :])), cmap='coolwarm')

                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    plt.savefig("./plot/mask_W{0}_h{1}_v{2}_lr{3}_batch{4}_p{5}_epoch{6}.png".format(method,n_hidden,n_visible,lr,batchsize,p,max_epoch), format="png")

                    fig = plt.figure()
                    cnt = 0
                    for i in range(n_hidden):
                        cnt += 1
                        ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                             int(np.sqrt(n_hidden)), cnt)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        im = plt.imshow(W_true.detach().numpy()[i, :].reshape([5, 5]),
                                        vmin = -max(abs(W_true.detach().numpy()[i, :])), 
                                        vmax = max(abs(W_true.detach().numpy()[i, :])), cmap='coolwarm')

                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    plt.savefig("./plot/true_W{0}_h{1}_v{2}_lr{3}_batch{4}_p{5}_epoch{6}.png".format(method,n_hidden,n_visible,lr,batchsize,p,max_epoch), format="png")

                    

                    fig = plt.figure()
                    cnt = 0
                    for i in range(n_hidden):
                        cnt += 1
                        ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                             int(np.sqrt(n_hidden)), cnt)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        im = plt.imshow(W_init.detach().numpy()[i, :].reshape([5, 5]),
                                        vmin = -max(abs(W_init.detach().numpy()[i, :])), 
                                        vmax = max(abs(W_init.detach().numpy()[i, :])), cmap='coolwarm')

                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    plt.savefig("./plot/init_W{0}_h{1}_v{2}_lr{3}_batch{4}_p{5}_epoch{6}.png".format(method,n_hidden,n_visible,lr,batchsize,p,max_epoch), format="png")





