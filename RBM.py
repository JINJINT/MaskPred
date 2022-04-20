
from utils import *

class RBM:
    """
    The RBM class
    """
    def __init__(self, n_visible, n_hidden, k, lr=0.01, minibatch_size=1, seed= 10417617, W_init=None, masked = False):
        """
        n_visible, n_hidden: dimension of visible and hidden layer
        k: number of gibbs sampling steps
        k: number of gibbs sampling steps
        lr: learning rate
        vbias, hbias: biases for visible and hidden layer, initialized as zeros
            in shape of (n_visible,) and (n_hidden,)
        W: weights between visible and hidden layer, initialized using Xavier,
            same as Assignment1-Problem1, in shape of (n_hidden, n_visible)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.seed = seed
        self.masked = masked
        
        self.vbias = torch.zeros(n_visible, requires_grad=True)
        self.hbias = torch.zeros(n_hidden, requires_grad=True)
        if W_init is not None:
            self.W = W_init
            self.W.requires_grad_(True)
        else:
            if self.seed is not None: 
                torch.manual_seed(self.seed)
            self.W = torch.empty((n_hidden, n_visible), requires_grad=True)
            torch.nn.init.normal_(self.W, 0, np.sqrt(6.0 / (self.n_hidden + self.n_visible)))
        

    def h_v(self, v, T=1):
        """
        Calculates hidden vector distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return P(h=1|v) in shape (N, n_hidden)
        N is the batch size
        """
        return sigmoid(self.hbias + T* v @ (self.W.T))

    def sample_h(self, v, T=1):
        """
        Sample hidden vector given distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return hidden vector and P(h=1|v) both in shape (N, n_hidden)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        
        return torch.tensor(np.random.binomial(1, self.h_v(v ,T=T)), dtype=torch.float32), self.h_v(v, T=T)

    def v_h(self, h, T=1):
        """
        Calculates visible vector distribution P(v=1|h)
        h: hidden vector in shape (N, n_hidden)
        return P(v=1|h) in shape (N, n_visible)
        """
        return sigmoid(self.vbias + T * h @ self.W)

    def sample_v(self, h, T=1):
        """
        Sample visible vector given distribution P(h=1|v)
        h: hidden vector in shape (N, n_hidden)
        return visible vector and P(v=1|h) both in shape (N, n_visible)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
    
        return torch.tensor(np.random.binomial(1, self.v_h(h, T=T)), dtype=torch.float32), self.v_h(h, T=T)

    def gibbs_k(self, v, k=0, T=1):
        """
        The contrastive divergence k (CD-k) procedure
        v: visible vector, in (N, n_visible)
        k: number of gibbs sampling steps
        return (h0, v0, h_sample, v_sample, prob_h, prob_v)
        h0: initial hidden vector sample, in (N, n_hidden)
        v0: the input v, in (N, n_visible)
        h_sample: the hidden vector sample after k steps, in (N, n_hidden)
        v_sample: the visible vector samplg after k steps, in (N, n_visible)
        prob_h: P(h=1|v) after k steps, in (N, n_hidden)
        prob_v: P(v=1|h) after k steps, in (N, n_visible)
        (Refer to Fig.1 in the handout if unsure on step counting)
        """
        
        with torch.no_grad():
            np.random.seed(self.seed)
            if (k==0): k=self.k
            v0 = torch.tensor(v, dtype=torch.float32)
            h0, prob_h = self.sample_h(v0, T=T)
            v_samples = []
            v_samples.append(v0)
            h_sample = h0
    
            for step in range(k):
                v_sample, prob_v = self.sample_v(h_sample, T=T)
                v_samples.append(v_sample)
                h_sample, prob_h = self.sample_h(v_sample, T=T)
        
        return v_samples


    def AIS(self, n,Jst,k,m):
        betalist = np.linspace(0,1,k+1)
        zhatlist = []
        model0 = RBM(n,0,0)

        for i in range(m):
            zhat=0
            model0.init_state()
            x = model._state
            for j in range(k):
              modelcur = (n,0,Jst*betalist[j+1])
              xcur=gibbs(modelcur,x,1)[0]
              zhat += Jst*sumJst(xcur,n)*(betalist[j+1]-betalist[j])
              x = xcur
            zhat += np.log(2)*n**2
            zhatlist.append(zhat)
        return np.mean(zhatlist), np.var(zhatlist)    


    def tp(self, p):
        T = np.linspace(0,1.0,20)
        Zlist = []
        for t in range(len(T)):
            Zlist.append(AIS(n, Jst/T[t],50,50)[0])

        sumlist = []
        templist = []  
        sumt1list = []
        sumgibbslist = []

        modelcur = Ising(n,0,0)
        x = modelcur._state
        t = 0
          
        modelgib = Ising(n,0,Jst)
        xgib = modelcur._state
          
        for i in range(100000):
            # Gibss sampling
            xgibcur = gibbs(modelgib,xgib,1)[0]
            sumgibbslist.append(np.sum(xgibcur))
            xgib = xgibcur.copy()
            
            # Tempering sampling
            if np.random.uniform(0,1)<=0.5:
              if t==0:
                tcur = 1
              elif t == len(T)-1:
                tcur = len(T)-2 
              else:
                if np.random.uniform(0,1)<=0.5:
                  tcur = t-1  
                else:
                  tcur = t+1  
              ratio = np.exp(Jst*sumJst(x,n)/T[tcur]-Zlist[tcur]-Jst*sumJst(x,n)/T[t]+Zlist[t])
              u = np.random.uniform(0,1)
              if u<=min(ratio,1):
                t = tcur
            else:
              modelcur = Ising(n,0,Jst/T[t])
              xcur = gibbs(modelcur,x,1)[0]
              x = xcur.copy()
            sumlist.append(np.sum(x))
            templist.append(T[t])
            
            if T[t]==1:
              sumt1list.append(np.sum(x))    

    
    def eval(self, X):
        """
        Computes reconstruction error, set k=1 for reconstruction.
        X: in (N, n_visible)
        Return the mean reconstruction error as scalar
        """
        sample_X = torch.stack(self.gibbs_k(X, k=1))[-1,]

        return torch.mean(torch.sqrt(torch.sum(torch.square(X - sample_X), axis=1)))

    def free_energy(self, X):
        """
        Computes free energy
        X: in (N, n_visible)
        Return the free energy as scalar
        """

        return -torch.mean(torch.sum(torch.log(torch.exp(self.hbias + X @ self.W.T)+1), axis=1)+ X @ self.vbias.T).detach().numpy() 
    


    def update(self, X, toupdate = True):
        """
        Updates RBM parameters with data X
        X: in (N, n_visible)
        Compute all gradients first before updating(/making change to) the
        parameters(W and biases).
        """
        
        loss = 0
        N = X.shape[0]
        n_visible = X.shape[1]

    
        if not self.masked:
            if toupdate:
                sample_X = torch.stack(self.gibbs_k(X,k=1))[-1,]
                expect_pv = self.h_v(sample_X)
                pv = self.h_v(X)

                expect_v = sample_X.clone()

                grad_hbias = expect_pv - pv
                grad_vbias = expect_v - X
                
                grad_W = expect_pv.T @ expect_v - pv.T @ X
                
                #self.hbias = self.hbias -self.lr *  torch.squeeze(grad_hbias)
                #self.vbias =  self.vbias - self.lr * torch.squeeze(grad_vbias)
                self.W =  self.W - self.lr * grad_W
            
            loss = -torch.mean(torch.sum(torch.log(torch.exp(self.hbias + X @ self.W.T)+1), axis=1)+ X @ self.vbias.T)

        else: 
            # masked loss    
            
            # Brute force
            '''
            for n in range(N):
                x = X[n]
                for i in range(n_visible):
                    x_temp1 = torch.clone(x)
                    x_temp1[i] = 1
                    x_temp0 = torch.clone(x)
                    x_temp0[i] = 0
                    log_p1 = torch.sum(torch.log(torch.exp(self.hbias + x_temp1 @ self.W.T)+1))+self.vbias[i]
                    log_p0 = torch.sum(torch.log(torch.exp(self.hbias + x_temp0 @ self.W.T)+1))
                    loss += x[i]*log_p1 + (1-x[i])*log_p0 - torch.log((torch.exp(log_p1) + torch.exp(log_p0)))
            loss = -loss / (N*n_visible)    
            loss.backward()
            '''
            # Matrix
            
            for i in range(n_visible):
                X_temp1 = torch.clone(X)
                X_temp1[:,i] = 1
                X_temp0 = torch.clone(X)
                X_temp0[:,i] = 0
                log_p1 = torch.sum(torch.log(torch.exp(self.hbias + X_temp1 @ self.W.T)+1), axis=1) + self.vbias[i]
                log_p0 = torch.sum(torch.log(torch.exp(self.hbias + X_temp0 @ self.W.T)+1), axis=1)
                loss += torch.mean(X[:,i]*log_p1 + (1-X[:,i])*log_p0 - torch.log(torch.exp(log_p1) + torch .exp(log_p0)))
            loss = -loss / n_visible
            
            if toupdate:
                loss.backward()
                
                with torch.no_grad():
                    #self.hbias -= self.lr *  self.hbias.grad
                    #self.vbias -= self.lr * self.vbias.grad
                    self.W -= self.lr * self.W.grad
                self.W.grad.zero_()
                self.vbias.grad.zero_()
                self.hbias.grad.zero_()

            return loss.detach()*N


    def train(self, X_train, X_valid, W_true, max_epoch=200, show = 10):

        n_train, n_visible = X_train.shape
        n_valid, n_visible = X_valid.shape
        training_loss = []
        valid_loss = []
        dist_W = []
        ratio = []
        err_train = []
        err_valid = []
        energy_train = []
        energy_valid = []

        for epoch in np.arange(max_epoch):
            train_loss = 0
            shuffle_indices = np.arange(0, n_train)
            np.random.shuffle(shuffle_indices)
            for i in range(n_train//self.minibatch_size):
                X_batch = X_train[shuffle_indices[i*self.minibatch_size:(i+1)*self.minibatch_size-1]]
                train_loss_batch = self.update(X_batch)
                train_loss += train_loss_batch
            
            training_loss.append(train_loss/n_train)
            dist_W.append(torch.norm(self.W.detach()-W_true.detach())/(self.n_hidden*self.n_visible))
            ratio.append(dist_W[-1]/dist_W[0])
            
            valid_loss.append(self.update(X_valid, toupdate = False)/n_valid)
            if not epoch % show:
                print('Epoch {0}'.format(epoch))
                print('train loss: {0}, valid loss: {1}, weight_diff: {2}, diff_ratio: {3}'.format(
                    training_loss[-1], valid_loss[-1], dist_W[-1], ratio[-1]))  

                err_train.append(self.eval(X_train))
                err_valid.append(self.eval(X_valid))
                energy_train.append(self.free_energy(X_train))
                energy_valid.append(self.free_energy(X_valid))

                print('err: Train {0}, Valid {1}'.format(
                    err_train[-1], err_valid[-1]))

                print('energy: Train {0}, Valid {1}'.format(
                    energy_train[-1], energy_valid[-1]))              
        
        return training_loss, valid_loss, dist_W, ratio, err_train, err_valid, energy_train, energy_valid

    def samp(self, n, check = False):
        v0 = torch.tensor(np.random.binomial(1, np.ones(self.n_visible)*0.5), dtype=torch.float32)
        samples = torch.stack(self.gibbs_k(v0))
        
        return samples
    




# if __name__ == "__main__":
#     np.seterr(all='raise')
#     parser = argparse.ArgumentParser(description='data, parameters, etc.')
#     parser.add_argument('-max_epoch', type=int, help="maximum epoch", default=300)
#     parser.add_argument('-k', type=int, help="CD-k sampling", default=1)
#     parser.add_argument('-lr', type=float, help="learning rate", default=0.02)
#     parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1)
#     parser.add_argument('-train', type=str, help='training file path', default='../data/digitstrain.txt')
#     parser.add_argument('-valid', type=str, help='validation file path', default='../data/digitsvalid.txt')
#     parser.add_argument('-test', type=str, help="test file path", default="../data/digitstest.txt")
#     parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=100)

#     args = parser.parse_args()

#     train_data = np.genfromtxt(args.train, delimiter=",")
#     train_X = train_data[:, :-1]
#     train_Y = train_data[:, -1]
#     train_X = binary_data(train_X)
#     valid_data = np.genfromtxt(args.valid, delimiter=",")
#     valid_X = valid_data[:, :-1]
#     valid_X = binary_data(valid_X)
#     valid_Y = valid_data[:, -1]
    
#     test_data = np.genfromtxt(args.test, delimiter=",")
#     test_X = test_data[:, :-1]
#     test_X = binary_data(test_X)
#     test_Y = test_data[:, -1]

#     """
#     Implement as you wish, not autograded
#     """

#     n_train, n_visible = train_X.shape

#     err_train = []
#     err_valid = []

#     model = RBM(n_visible, n_hidden=args.n_hidden, k=args.k, lr=args.lr,
#          minibatch_size=args.minibatch_size, seed = None)
    
#     for epoch in np.arange(0, args.max_epoch):
        
#         shuffle_indices = np.arange(0, n_train)
#         np.random.shuffle(shuffle_indices)
        
#         for idx in shuffle_indices:
#             model.update(train_X[idx, :].reshape((1,-1)))

#         err_train.append(model.eval(train_X))
#         err_valid.append(model.eval(valid_X))

#         if not epoch % 10:
#             print('Epoch {0}'.format(epoch))
#             print('err: Train {0}, Valid {1}'.format(
#                 err_train[-1], err_valid[-1]))        
    
    

#     # Plot result
#     plt.figure()
#     plt.plot(range(args.max_epoch), err_train, label='Train')
#     plt.plot(range(args.max_epoch), err_valid,label='Validation')
#     plt.xlabel('Epoch')
#     plt.ylabel('Reconstruction Error')
#     plt.legend()
#     plt.savefig(
#         "../plot/RBM1_k{0}_err.png".format(model.k), format="png")

    # # Plot W
    # fig = plt.figure()
    # cnt = 0
    # for i in range(args.n_hidden):
    #     cnt += 1
    #     ax = fig.add_subplot(int(np.sqrt(args.n_hidden)),
    #                          int(np.sqrt(args.n_hidden)), cnt)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     im = plt.imshow(model.W[i, :].reshape([28, 28]))

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.savefig("../plot/RBM1_k{0}_W.png".format(model.k), format="png")


    # Plot test original
    # np.random.seed(seed)
    # idxtest = np.random.randint(0, test_data.shape[0], size=100)
    # fig = plt.figure()
    # cnt = 0
    # for i in idxtest:
    #     cnt += 1
    #     ax = fig.add_subplot(int(np.sqrt(100)),
    #                          int(np.sqrt(100)), cnt)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     im = plt.imshow(test_X[i, :].reshape([28, 28]), cmap='gray')

    # plt.savefig(
    #     "../plot/RBM1_k{0}_original_test.png".format(model.k), format="png")
    
    
    # # Plot test reconstruct
    # _, _, _, sample_X, _, _ = model.gibbs_k(test_X[idxtest, :])
    # fig = plt.figure()
    # cnt = 0
    # for i in range(100):
    #     cnt += 1
    #     ax = fig.add_subplot(int(np.sqrt(100)),
    #                          int(np.sqrt(100)), cnt)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     im = plt.imshow(sample_X[i,:].reshape([28, 28]), cmap='gray')

    # plt.savefig("../plot/RBM1_k{0}_recons_test.png".format(model.k), format="png")


    

