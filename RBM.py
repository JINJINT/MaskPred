
from utils import *
from RBMucd import *

class RBM:
    """
    The RBM class
    """
    def __init__(self, n_visible, n_hidden, k, lr=0.01, minibatch_size=1, 
                 seed= 10417617, W_init=None, method = 'cd',
                 sparsity=None, lbd=0.1, 
                 rand = 0.2, max_epoch = 1000):
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
        self.sparsity = sparsity
        self.lbd = lbd
        self.rand = rand # in (0,1)
        self.method = method
        self.max_epoch = max_epoch
                
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

        if self.method=='UCD':
            self.ucd = RBMucd(n_visible, n_hidden, 
                              w0 = nd.array(self.W.detach().numpy().T),
                              b0 = nd.zeros(n_visible),
                              c0 = nd.zeros(n_hidden))

        

    def h_v(self, v):
        """
        Calculates hidden vector distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return P(h=1|v) in shape (N, n_hidden)
        N is the batch size
        """
        return sigmoid(self.hbias + v @ (self.W.T))

    def sample_h(self, v):
        """
        Sample hidden vector given distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return hidden vector and P(h=1|v) both in shape (N, n_hidden)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        
        return torch.tensor(np.random.binomial(1, self.h_v(v)), dtype=torch.float32), self.h_v(v)

    def v_h(self, h):
        """
        Calculates visible vector distribution P(v=1|h)
        h: hidden vector in shape (N, n_hidden)
        return P(v=1|h) in shape (N, n_visible)
        """
        return sigmoid(self.vbias + h @ self.W)

    def sample_v(self, h):
        """
        Sample visible vector given distribution P(h=1|v)
        h: hidden vector in shape (N, n_hidden)
        return visible vector and P(v=1|h) both in shape (N, n_visible)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
    
        return torch.tensor(np.random.binomial(1, self.v_h(h)), dtype=torch.float32), self.v_h(h)

    def gibbs_k(self, v, k=0):
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
            v0 = v.clone()
            h0, prob_h = self.sample_h(v0)
            v_samples = []
            v_samples.append(v0)
            h_sample = h0
    
            for step in range(k):
                v_sample, prob_v = self.sample_v(h_sample)
                v_samples.append(v_sample)
                h_sample, prob_h = self.sample_h(v_sample)
        
        return v_samples
 

    
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
    


    def update(self, X, ep = 1, toupdate = True):
        """
        Updates RBM parameters with data X
        X: in (N, n_visible)
        Compute all gradients first before updating(/making change to) the
        parameters(W and biases).
        """
        
        loss = 0
        N = X.shape[0]
        n_visible = X.shape[1]
        
        if self.method=='CD': # CD-k algorithm
            penalty = 0

            if toupdate:
                sample_X = torch.stack(self.gibbs_k(X, k=1))[-1,]
                expect_pv = self.h_v(sample_X)
                pv = self.h_v(X) # (N, n_hidden)

                expect_v = sample_X.clone()

                grad_hbias = expect_pv - pv
                grad_vbias = expect_v - X
                
                grad_W = (expect_pv.T @ expect_v - pv.T @ X)/N 

                # if self.sparsity is not None:
                #     grad_W =  grad_W  + self.lbd*(torch.mean(pv - self.sparsity, dim=0).reshape((1,-1)).T @ torch.sum(X, dim=0).reshape((1,n_visible)))/N
                #     penalty = penalty - torch.sum(torch.squeeze(self.sparsity * torch.log(torch.mean(pv, dim=0)) + (1-self.sparsity) * torch.log(1-torch.mean(pv, dim=0))))

                #self.hbias = self.hbias -self.lr *  torch.mean(torch.squeeze(grad_hbias), dim=0) - torch.mean(pv - self.sparsity, dim=0) 
                #self.vbias =  self.vbias - self.lr * torch.mean(torch.squeeze(grad_vbias), dim=0)
                self.W =  self.W - self.lr /np.power(ep,1/2) * grad_W
                if self.sparsity is not None:
                    thred = torch.quantile(torch.abs(torch.flatten(self.W)), q=1-2*self.sparsity)
                    self.W = self.W * (torch.nn.functional.relu(torch.abs(self.W) - thred)>0)

            loss = torch.mean(torch.sum(torch.log(torch.exp(self.hbias + X @ self.W.T)+1), dim=1)+ X @ self.vbias.T)  
            if self.sparsity is not None:
                loss = loss + torch.sum(torch.abs(self.W))

        if self.method=='UCD': # unbiased CD-k algorithm
            if ep < self.max_epoch*0.7:
                if toupdate:
                    sample_X = torch.stack(self.gibbs_k(X, k=1))[-1,]
                    expect_pv = self.h_v(sample_X)
                    pv = self.h_v(X) # (N, n_hidden)

                    expect_v = sample_X.clone()

                    grad_hbias = expect_pv - pv
                    grad_vbias = expect_v - X
                    
                    grad_W = (expect_pv.T @ expect_v - pv.T @ X)/N 

                    # if self.sparsity is not None:
                    #     grad_W =  grad_W  + self.lbd*(torch.mean(pv - self.sparsity, dim=0).reshape((1,-1)).T @ torch.sum(X, dim=0).reshape((1,n_visible)))/N
                    #     penalty = penalty - torch.sum(torch.squeeze(self.sparsity * torch.log(torch.mean(pv, dim=0)) + (1-self.sparsity) * torch.log(1-torch.mean(pv, dim=0))))

                    #self.hbias = self.hbias -self.lr *  torch.mean(torch.squeeze(grad_hbias), dim=0) - torch.mean(pv - self.sparsity, dim=0) 
                    #self.vbias =  self.vbias - self.lr * torch.mean(torch.squeeze(grad_vbias), dim=0)
                    self.W =  self.W - self.lr /np.power(ep,1/2) * grad_W
                    if self.sparsity is not None:
                        thred = torch.quantile(torch.abs(torch.flatten(self.W)), q=1-2*self.sparsity)
                        self.W = self.W * (torch.nn.functional.relu(torch.abs(self.W) - thred)>0)

                    self.ucd = RBMucd(self.n_visible, self.n_hidden, 
                              w0 = nd.array(self.W.detach().numpy().T),
                              b0 = nd.zeros(self.n_visible),
                              c0 = nd.zeros(self.n_hidden))    

            else:
                if toupdate:
                    dat = nd.array(X)
                    self.ucd.compute_grad1(dat)
                    self.ucd.zero_grad2()
                    for j in range(100):
                        tau_t, disc_t = self.ucd.accumulate_grad2_ucd(dat, min_mcmc=1, max_mcmc=100)
                    self.ucd.update_param(self.lr, nchain=100)
                    self.W =  torch.tensor(self.ucd.w.asnumpy().T.copy())
                    if self.sparsity is not None:
                        thred = torch.quantile(torch.abs(torch.flatten(self.W)), q=1-2*self.sparsity)
                        self.W = self.W * (torch.nn.functional.relu(torch.abs(self.W) - thred)>0)

            loss = torch.mean(torch.sum(torch.log(torch.exp(self.hbias + X @ self.W.T)+1), dim=1)+ X @ self.vbias.T) 
            if self.sparsity is not None:
                loss = loss + torch.sum(torch.abs(self.W))


        if self.method=='pseudo': # pseudolikelihood loss
            for i in range(n_visible):
                X_temp1 = torch.clone(X)
                X_temp1[:,i] = 1
                X_temp0 = torch.clone(X)
                X_temp0[:,i] = 0
                log_p1 = torch.sum(torch.log(torch.exp(self.hbias + X_temp1 @ self.W.T)+1), axis=1) + self.vbias[i]
                log_p0 = torch.sum(torch.log(torch.exp(self.hbias + X_temp0 @ self.W.T)+1), axis=1)
                loss += torch.mean(X[:,i]*log_p1 + (1-X[:,i])*log_p0 - torch.log(torch.exp(log_p1) + torch .exp(log_p0)))
            loss = -loss / n_visible
            if self.sparsity is not None:
                loss = loss +  0.001*torch.sum(torch.abs(self.W))
            
            if toupdate:
                loss.backward()
                
                with torch.no_grad():
                    #self.hbias -= self.lr *  self.hbias.grad
                    #self.vbias -= self.lr * self.vbias.grad
                    self.W -= self.lr /np.power(ep,1/2) * self.W.grad
  
                self.W.grad.zero_()
                self.vbias.grad.zero_()
                self.hbias.grad.zero_()


        if self.method=='randmask': # random masking loss
            rand_indices = sample(range(n_visible), math.ceil(n_visible * self.rand))
            for i in rand_indices:
                X_temp1 = torch.clone(X)
                X_temp1[:,i] = 1
                X_temp0 = torch.clone(X)
                X_temp0[:,i] = 0
                log_p1 = torch.sum(torch.log(torch.exp(self.hbias + X_temp1 @ self.W.T)+1), axis=1) + self.vbias[i]
                log_p0 = torch.sum(torch.log(torch.exp(self.hbias + X_temp0 @ self.W.T)+1), axis=1)
                loss += torch.mean(X[:,i]*log_p1 + (1-X[:,i])*log_p0 - torch.log(torch.exp(log_p1) + torch .exp(log_p0)))
            loss = -loss / n_visible
            if self.sparsity is not None:
                loss = loss +  0.001*torch.sum(torch.abs(self.W))
            
            if toupdate:
                loss.backward()
                
                with torch.no_grad():
                    #self.hbias -= self.lr *  self.hbias.grad
                    #self.vbias -= self.lr * self.vbias.grad
                    self.W -= self.lr /np.power(ep,1/2)* self.W.grad
  
                self.W.grad.zero_()
                self.vbias.grad.zero_()
                self.hbias.grad.zero_()

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

        return loss.detach()*N


    def train(self, X_train, X_valid, W_true, show = 10):

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

        for epoch in np.arange(self.max_epoch):
            train_loss = 0
            shuffle_indices = np.arange(0, n_train)
            np.random.shuffle(shuffle_indices)

            dist_W.append(torch.norm(self.W.detach()-W_true.detach())/(self.n_hidden*self.n_visible))
            ratio.append(dist_W[-1]/dist_W[0])
            for i in range(n_train//self.minibatch_size):
                X_batch = X_train[shuffle_indices[i*self.minibatch_size:(i+1)*self.minibatch_size-1]]
                train_loss_batch = self.update(X_batch, ep = epoch+1)
                train_loss += train_loss_batch
            training_loss.append(train_loss/n_train)
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
        
        self.training_loss = training_loss
        self.valid_loss = valid_loss
        self.dist_W = dist_W
        self.ratio = ratio
        self.err_train = err_train
        self.err_valid = err_valid
        self.energy_train = energy_train
        self.energy_valid = energy_valid
        return training_loss, valid_loss, dist_W, ratio, err_train, err_valid, energy_train, energy_valid

    def samp(self, n, check = False):
        v0 = torch.tensor(np.random.binomial(1, np.ones(self.n_visible)*0.5), dtype=torch.float32)
        samples = torch.stack(self.gibbs_k(v0))
        
        return samples

    





