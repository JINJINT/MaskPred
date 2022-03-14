from utils import *

class RBM_base:
    """
    The RBM base class
    """
    def __init__(self, n_visible, n_hidden, k, W_init, lr=0.01, minibatch_size=1, seed= 10417617):
        """
        n_visible, n_hidden: dimension of visible and hidden layer
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
        
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)
        if self.seed is not None: 
            np.random.seed(self.seed)
        
        if W_init is not None:
            self.W = W_init.copy()
        else:
            self.W = np.random.normal(0, np.sqrt(
            6.0 / (self.n_hidden + self.n_visible)), (n_hidden, n_visible))
        

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
        if self.seed is not None: 
            np.random.seed(self.seed)
        return np.random.binomial(1, self.h_v(v)), self.h_v(v)

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
        if self.seed is not None: 
            np.random.seed(self.seed)
        return np.random.binomial(1, self.v_h(h)), self.v_h(h)

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

        if (k==0): k=self.k
        v0 = v
        h0, prob_h = self.sample_h(v)
        prob_v = np.zeros_like(v)

        v_sampple = v0.copy()
        h_sample = h0.copy()

        for step in range(k):
            v_sample, prob_v = self.sample_v(h_sample)
            h_sample, prob_h = self.sample_h(v_sample)

        return h0, v0, h_sample, v_sample, prob_h, prob_v    


    def update(self, X, toupdate = True):
        """
        Updates RBM parameters with data X
        X: in (N, n_visible)
        Compute all gradients first before updating(/making change to) the
        parameters(W and biases).
        """
        N,_ = X.shape

        if toupdate:
            _, _, _, sample_X, _, _ = self.gibbs_k(X)
            expect_pv = self.h_v(sample_X)
            pv = self.h_v(X)
            expect_v = sample_X.copy()

            grad_hbias = expect_pv - pv
            grad_vbias = expect_v - X
            grad_W = (expect_pv.T @ expect_v - pv.T @ X)/N

            
            self.hbias -= self.lr *  np.mean(np.squeeze(grad_hbias),axis=0)
            self.vbias -= self.lr * np.mean(np.squeeze(grad_vbias),axis=0)
            self.W -= self.lr * grad_W
                

        return np.mean(np.sum(np.log(np.exp(self.hbias + X @ self.W.T)+1), axis=1)+ X @ self.vbias.T)*N

    def eval(self, X):
        """
        Computes reconstruction error, set k=1 for reconstruction.
        X: in (N, n_visible)
        Return the mean reconstruction error as scalar
        """
        _, _, _, sample_X, _, _ = self.gibbs_k(X, k=1)

        return np.average(np.sqrt(np.sum(np.square(X - sample_X), axis=1)))

    def free_engery(self, X):
        """
        Computes free energt
        X: in (N, n_visible)
        Return the free energy as scalar
        """

        return -np.mean(np.sum(np.log(np.exp(self.hbias + X @ self.W.T)+1), axis=1)+ X @ self.vbias.T)    


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
        
            shuffle_indices = np.arange(0, n_train)
            np.random.shuffle(shuffle_indices)

            train_loss = 0
            shuffle_indices = np.arange(0, n_train)
            np.random.shuffle(shuffle_indices)
            for i in range(n_train//self.minibatch_size):
                X_batch = X_train[shuffle_indices[i*self.minibatch_size:(i+1)*self.minibatch_size-1]]
                
                train_loss_batch = self.update(X_batch)
                train_loss += train_loss_batch
                

            training_loss.append(train_loss/n_train)
            valid_loss.append(self.update(X_valid, toupdate = False)/n_valid)
            dist_W.append(norm(self.W-W_true)/(self.n_hidden*self.n_visible))
            ratio.append(dist_W[-1]/dist_W[0])

            if not epoch % show:
                print('Epoch {0}'.format(epoch))
                print('train loss: {0}, test loss: {1}, weight_diff: {2}, diff_ratio: {3}'.format(
                    training_loss[-1], valid_loss[-1], dist_W[-1], ratio[-1]))     
                
                err_train.append(self.eval(X_train))
                err_valid.append(self.eval(X_valid))
                energy_train.append(self.free_engery(X_train))
                energy_valid.append(self.free_engery(X_valid))
                print('error: Train {0}, Test {1}'.format(
                    err_train[-1], err_valid[-1]))
                print('energy: Train {0}, Valid {1}'.format(
                    energy_train[-1], energy_valid[-1]))

        return training_loss, valid_loss, dist_W, ratio, err_train, err_valid, energy_train, energy_valid
    
        

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
    
#     with open('../dump/RBM_k{0}.pickle'.format(args.k), 'wb') as handle:
#         pickle.dump(model, handle)

#     # Check Reconstruction from test set
#     with open('../dump/RBM_k{0}.pickle'.format(args.k), "rb") as handle:
#         model = pickle.load(handle)

#     # Plot result
#     plt.figure()
#     plt.plot(range(args.max_epoch), err_train, label='Train')
#     plt.plot(range(args.max_epoch), err_valid,label='Validation')
#     plt.xlabel('Epoch')
#     plt.ylabel('Reconstruction Error')
#     plt.legend()
#     plt.savefig(
#         "../plot/RBM1_k{0}_err.png".format(model.k), format="png")

#     # Plot W
#     fig = plt.figure()
#     cnt = 0
#     for i in range(args.n_hidden):
#         cnt += 1
#         ax = fig.add_subplot(int(np.sqrt(args.n_hidden)),
#                              int(np.sqrt(args.n_hidden)), cnt)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         im = plt.imshow(model.W[i, :].reshape([28, 28]))

#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     plt.savefig("../plot/RBM1_k{0}_W.png".format(model.k), format="png")


#     # Plot test original
#     np.random.seed(seed)
#     idxtest = np.random.randint(0, test_data.shape[0], size=100)
#     fig = plt.figure()
#     cnt = 0
#     for i in idxtest:
#         cnt += 1
#         ax = fig.add_subplot(int(np.sqrt(100)),
#                              int(np.sqrt(100)), cnt)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         im = plt.imshow(test_X[i, :].reshape([28, 28]), cmap='gray')

#     plt.savefig(
#         "../plot/RBM1_k{0}_original_test.png".format(model.k), format="png")
    
    
#     # Plot test reconstruct
#     _, _, _, sample_X, _, _ = model.gibbs_k(test_X[idxtest, :])
#     fig = plt.figure()
#     cnt = 0
#     for i in range(100):
#         cnt += 1
#         ax = fig.add_subplot(int(np.sqrt(100)),
#                              int(np.sqrt(100)), cnt)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         im = plt.imshow(sample_X[i,:].reshape([28, 28]), cmap='gray')

#     plt.savefig("../plot/RBM1_k{0}_recons_test.png".format(model.k), format="png")


    

