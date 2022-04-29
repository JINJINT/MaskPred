from RBM import *
from utils import *
from RBMucd import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns   

if not os.path.exists('./plot'):
    os.makedirs('./plot')

if not os.path.exists('./dump'):
    os.makedirs('./dump')    
    
# parameters 
n_hidden = 9
n_visible = 25
n_train = 1000
n_valid = 100
max_epoch = 2000
k = 10
show = 10
plotw = False

for pattern in ['sparse']:
    for noise in [0.5]:      
        for p in [0.1,None]:
            for batchsize in [100]:
                        #======generate true weights
                W_true = generate_W(pattern, n_hidden, n_visible, p=0.1, seed=521)
                n_hidden, n_visible = W_true.shape
                
                for method in ['CD','pseudo','randmask','UCD']:
                    if method == 'CD':
                        lr = noise*0.05*2
                        max_epoch = 2000
                    if method == 'randmask':
                        lr = noise*0.5*2
                        max_epoch = 2000
                    if method == 'pseudo':
                        lr = noise*0.5*2
                        max_epoch = 2000
                    if method =='UCD':
                        lr = noise*0.1
                        max_epoch = 200
    

                    train_loss = []
                    val_loss = []
                    wdiff = []
                    wratio= []   
                    train_energy = []
                    val_energy = [] 

                    for trial in range(10):    
                        seed = 521 + trial
                        # Generate initial weights
                        # warm start
                        if pattern=='random':    
                            W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                        noise *6 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                        
                        if pattern=='sparse':
                            W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                        noise *6 / (n_hidden + n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                        
                        if pattern=='real':
                            W_init = W_true + torch.tensor(np.random.normal(0, np.sqrt(
                                        noise*1*min(torch.norm(W_true),6)/(n_hidden*n_visible)), (n_hidden, n_visible)), dtype=torch.float32)
                                   

                        # Generate data from W_true
                        rbm_true =  RBM(n_visible, n_hidden, n_train + n_valid + 5000, lr, batchsize, seed=seed, W_init = torch.clone(W_true))
                        samples = rbm_true.samp(n_train + 5000)
                        samples_converged = samples[5001:,]
                        shuffle_indices = np.arange(0, n_train + n_valid)
                        np.random.shuffle(shuffle_indices)
                        samples_train = samples_converged[shuffle_indices[:n_train],]
                        samples_valid = samples_converged[shuffle_indices[n_train:],]

                        if plotw:
                            fig = plt.figure()
                            cnt = 0
                            for i in range(n_hidden):
                                cnt += 1
                                ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                                     int(np.sqrt(n_hidden)), cnt)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                im = plt.imshow(W_true.detach().numpy()[cnt-1, :].reshape((5, 5)),
                                                vmin = -1, 
                                                vmax = 1, cmap='coolwarm')

                            fig.subplots_adjust(right=0.8)
                            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                            fig.colorbar(im, cax=cbar_ax)
                            plt.savefig("./plot/true_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.png".format(pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), format="png")


                            fig = plt.figure()
                            cnt = 0
                            for i in range(n_hidden):
                                cnt += 1
                                ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                                     int(np.sqrt(n_hidden)), cnt)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                im = plt.imshow(W_init.detach().numpy()[i, :].reshape((5, 5)),
                                                vmin = -1, #max(abs(W_init.detach().numpy()[i, :])
                                                vmax = 1, cmap='coolwarm')

                            fig.subplots_adjust(right=0.8)
                            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                            fig.colorbar(im, cax=cbar_ax)
                            plt.savefig("./plot/init_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.png".format(pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), format="png")

                     
      

                        if not os.path.isfile('./dump/RBM_method{}_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.pickle'.format(method,pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial)):
                            rbm_train = RBM(n_visible, n_hidden, k, W_init=torch.clone(W_init), 
                                                 sparsity = p, lr=lr, method = method, 
                                                 minibatch_size=batchsize, seed= seed,
                                                 max_epoch = max_epoch)
                            training_loss, valid_loss, dist_weights, ratio, err_train, err_valid, energy_train, energy_valid = rbm_train.train(samples_train, samples_valid, W_true, show)
                            
                            with open('./dump/RBM_method{}_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.pickle'.format(method,pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), 'wb') as handle:
                                pickle.dump(rbm_train, handle)   
                        else:
                            with open('./dump/RBM_method{}_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.pickle'.format(method,pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), 'rb') as handle:
                                rbm_train = pickle.load(handle)  

                            training_loss =  rbm_train.training_loss
                            valid_loss =  rbm_train.valid_loss
                            dist_weights =  rbm_train.dist_W
                            ratio =  rbm_train.ratio
                            energy_train =  rbm_train.energy_train    
                            energy_valid =  rbm_train.energy_valid
                        
                        train_loss.append(training_loss)
                        val_loss.append(valid_loss) 
                        wdiff.append(dist_weights) 
                        wratio.append(ratio)
                        train_energy.append(energy_train)
                        val_energy.append(energy_valid)

                        if plotw:
                            # plot W
                            fig = plt.figure()
                            cnt = 0
                            for i in range(n_hidden):
                                cnt += 1
                                ax = fig.add_subplot(int(np.sqrt(n_hidden)),
                                                     int(np.sqrt(n_hidden)), cnt)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                im = plt.imshow(rbm_train.W.detach().numpy()[i, :].reshape([5, 5]),
                                                vmin = -1, 
                                                vmax = 1, 
                                                cmap='coolwarm')

                        
                            fig.subplots_adjust(right=0.8)
                            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                            fig.colorbar(im, cax=cbar_ax)
                            plt.savefig("./plot/W_method{}_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}_trial{}.png".format(method,pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), format="png")

                    
                    fig, (ax1, ax3, ax4, ax2) = plt.subplots(nrows=1, ncols=4, figsize=(12,3))
                    
                    trainloss = np.array(train_loss)
                    trainloss_mean = np.mean(trainloss, axis=0)
                    trainloss_std = np.std(trainloss, axis=0)
                
                    trainloss_ub = trainloss_mean + trainloss_std
                    trainloss_lb = trainloss_mean - trainloss_std

                    valloss = np.array(val_loss)
                    valloss_mean = np.mean(valloss, axis=0)
                    valloss_std = np.std(valloss, axis=0)
                
                    valloss_ub = valloss_mean + valloss_std
                    valloss_lb = valloss_mean - valloss_std

                    ax1.plot(np.arange(len(trainloss_mean)), trainloss_mean, label = 'Train',alpha=.6)
                    ax1.fill_between(np.arange(len(trainloss_ub)), trainloss_ub, trainloss_lb, alpha=.2, color = 'royalblue')
                    ax1.plot(np.arange(len(valloss_mean)), valloss_mean, label = 'Validation',alpha=.6)
                    ax1.fill_between(np.arange(len(valloss_ub)), valloss_ub, valloss_lb, alpha=.2, color='orange')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('{} loss'.format(method))
                    ax1.legend()


                    wdif = np.array(wdiff)
                    wdif_mean = np.mean(wdif, axis=0)
                    wdif_std = np.std(wdif, axis=0)
                
                    wdif_ub = wdif_mean + wdif_std
                    wdif_lb = wdif_mean - wdif_std

                    ax2.plot(np.arange(len(wdif_mean)), wdif_mean)
                    ax2.fill_between(np.arange(len(wdif_ub)), wdif_ub, wdif_lb, alpha=.2)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Weight difference')
                    ax2.set_title('{} weight diff'.format(method))
                    
                    trainenergy = np.array(train_energy)
                    trainenergy_mean = np.mean(trainenergy, axis=0)
                    trainenergy_std = np.std(trainenergy, axis=0)
                
                    trainenergy_ub = trainenergy_mean + trainenergy_std
                    trainenergy_lb = trainenergy_mean - trainenergy_std

                    valenergy = np.array(val_energy)
                    valenergy_mean = np.mean(valenergy, axis=0)
                    valenergy_std = np.std(valenergy, axis=0)
                
                    valenergy_ub = valenergy_mean + valenergy_std
                    valenergy_lb = valenergy_mean - valenergy_std

                    ax3.plot(show*np.arange(len(trainenergy_mean)), trainenergy_mean, label = 'Train')
                    ax3.fill_between(show*np.arange(len(trainenergy_ub)), trainenergy_ub, trainenergy_lb, alpha=.2, color = 'blue')
                    ax3.plot(show*np.arange(len(valenergy_mean)), valenergy_mean, label = 'Validation')
                    ax3.fill_between(show*np.arange(len(valenergy_ub)), valenergy_ub, valenergy_lb, alpha=.2, color = 'orange')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Free energy')
                    ax3.set_title('{} energy'.format(method))
                    ax3.legend()


                    wrat = np.array(wratio)
                    wrat_mean = np.mean(wrat, axis=0)
                    wrat_std = np.std(wdif, axis=0)
                
                    wrat_ub = wrat_mean + wrat_std
                    wrat_lb = wrat_mean - wrat_std

                    ax4.plot(np.arange(len(training_loss)), wrat_mean)
                    ax4.fill_between(np.arange(len(wrat_ub)), wrat_ub, wrat_lb, alpha=.2)
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Ratio of weight difference')
                    ax4.set_ylim(bottom=0)
                    ax4.set_title('{} weight diff (relative)'.format(method))

                    fig.tight_layout()
                    plt.savefig("./plot/RBM_method{}_pattern{}_h{}_v{}_lr{}_batch{}_p{}_epoch{}_noise{}.png".format(method,pattern,n_hidden,n_visible,lr,batchsize,p,max_epoch,noise,trial), format="png")
                        
                        
                    
       

