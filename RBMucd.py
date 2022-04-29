from utils import *

class RBMSampler:
    # Constructor
    # w [m x n], b [m], c [n]
    def __init__(self, w, b, c, ctx=mx.cpu()):
        if len(w.shape) != 2:
            raise ValueError("w must be a 2-d array")
        if len(b.shape) != 1 or len(c.shape) != 1:
            raise ValueError("b and c must be 1-d arrays")

        self.ctx = ctx
        self.m = w.shape[0]
        self.n = w.shape[1]
        self.w = w.as_in_context(self.ctx)
        self.b = b.as_in_context(self.ctx)
        self.c = c.as_in_context(self.ctx)

    # Sample v given h
    # h [N x n]
    def sample_v_given_h(self, h):
        vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
        v = nd.random.uniform(shape=vmean.shape, ctx=self.ctx) <= vmean
        return v

    # Sample h given v
    # v [N x m]
    def sample_h_given_v(self, v):
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        h = nd.random.uniform(shape=hmean.shape, ctx=self.ctx) <= hmean
        return h

    # Gibbs with k steps
    # v0 [N x m]
    def sample_k(self, v0, k):
        h = self.sample_h_given_v(v0)
        for i in range(k):
            v = self.sample_v_given_h(h)
            h = self.sample_h_given_v(v)
        return v, h



class UnbiasedRBMSampler:
    # Constructor
    # w [m x n], b [m], c [n]
    def __init__(self, w, b, c, ctx=mx.cpu()):
        if len(w.shape) != 2:
            raise ValueError("w must be a 2-d array")
        if len(b.shape) != 1 or len(c.shape) != 1:
            raise ValueError("b and c must be 1-d arrays")

        self.ctx = ctx
        self.m = w.shape[0]
        self.n = w.shape[1]
        self.w = w.as_in_context(self.ctx)
        self.b = b.as_in_context(self.ctx)
        self.c = c.as_in_context(self.ctx)

    def clip_sigmoid(self, x):
        return nd.sigmoid(nd.clip(x, -10.0, 10.0))

    # Sample from Bernoulli distribution
    def sample_bernoulli(self, prob):
        return nd.random.uniform(shape=prob.shape, ctx=self.ctx) <= prob

    # Sample v given h
    # h [N x n]
    def sample_v_given_h(self, h):
        vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
        return self.sample_bernoulli(vmean)

    # Sample h given v
    # v [N x m]
    def sample_h_given_v(self, v):
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        return self.sample_bernoulli(hmean)

    # (xi1, eta0) -> (xi2, eta1) -> ...
    # xi = (v, h), eta = (vc, hc)
    def max_coup(self, vc0, hc0, v1, h1, max_try=10):
        # Sample the xi chain
        # p(v | h1)
        v2mean = self.clip_sigmoid(nd.dot(self.w, h1) + self.b)
        v2 = self.sample_bernoulli(v2mean)
        # p(h | v)
        h2 = self.sample_h_given_v(v2)

        # If xi1 == eta0, also make xi2 == eta1 and early exit
        if nd.norm(v1 - vc0).asscalar() == 0 and nd.norm(h1 - hc0).asscalar() == 0:
            vc1 = v2.copy()
            hc1 = h2.copy()
            return vc1, hc1, v2, h2, 0

        # Let the two chains meet with a positive probability
        # p((v, h) | xi1) = p1(v | h1) * p2(h | v)
        # p((v, h) | eta0) = q1(v | hc0) * q2(h | v)
        # p2 = q2, so p((v, h) | xi1) / p((v, h) | eta0) = p1(v | h1) / q1(v | hc0)
        vc1mean = self.clip_sigmoid(nd.dot(self.w, hc0) + self.b)
        logpxi1 = nd.sum(nd.log(v2mean) * v2 + nd.log(1 - v2mean) * (1 - v2)).asscalar()
        logpeta0 = nd.sum(nd.log(vc1mean) * v2 + nd.log(1 - vc1mean) * (1 - v2)).asscalar()
        u = nd.random.exponential().asscalar()
        if u >= logpxi1 - logpeta0:
            vc1 = v2.copy()
            hc1 = h2.copy()
            return vc1, hc1, v2, h2, 0

        # Otherwise, sample the two chains conditional on no-meet
        v2 = None
        vc1 = None
        for i in range(max_try):
            # Common RNG
            uv = nd.random.uniform(shape=self.m, ctx=self.ctx)
            # Sample v2
            if v2 is None:
                v2 = uv <= v2mean
                # Accept v2 with probability 1-q(v2)/p(v2)
                # <=> Exp(1) < log[p(v2)] - log[q(v2)]
                logpv2 = nd.sum(nd.log(v2mean) * v2 + nd.log(1 - v2mean) * (1 - v2)).asscalar()
                logqv2 = nd.sum(nd.log(vc1mean) * v2 + nd.log(1 - vc1mean) * (1 - v2)).asscalar()
                u1 = nd.random.exponential().asscalar()
                if i < max_try - 1 and u1 >= logpv2 - logqv2:
                    v2 = None
            # Sample vc1
            if vc1 is None:
                vc1 = uv <= vc1mean
                # Accept vc1 with probability 1-p(vc1)/q(vc1)
                # <=> Exp(1) < log[q(vc1)] - log[p(vc1)]
                logpvc1 = nd.sum(nd.log(v2mean) * vc1 + nd.log(1 - v2mean) * (1 - vc1)).asscalar()
                logqvc1 = nd.sum(nd.log(vc1mean) * vc1 + nd.log(1 - vc1mean) * (1 - vc1)).asscalar()
                u2 = nd.random.exponential().asscalar()
                if i < max_try - 1 and u2 >= logqvc1 - logpvc1:
                    vc1 = None
            # Exit if v2 and vc1 have been set
            if v2 is not None and vc1 is not None:
                break

        # Sample h
        uh = nd.random.uniform(shape=self.n, ctx=self.ctx)
        h2mean = self.clip_sigmoid(nd.dot(self.w.T, v2) + self.c)
        hc1mean = self.clip_sigmoid(nd.dot(self.w.T, vc1) + self.c)
        h2 = uh <= h2mean
        hc1 = uh <= hc1mean

        return vc1, hc1, v2, h2, i

    # Unbiased sampling
    def sample(self, v0, min_steps=1, max_steps=100):
        # (v0, h0)   -> (v1, h1)   -> (v2, h2)   -> ... -> (vt, ht)
        # (vc0, hc0) -> (vc1, hc1) -> ... -> (vct, hct)
        # Init: (v0, h0) = (vc0, hc0)
        # Iter: (v1, h1, vc0, hc0) -> (v2, h2, vc1, hc1) -> ...
        # Stop: (vt, ht) = (vct, hct)
        vc = v0
        hc = self.sample_h_given_v(vc)
        v = self.sample_v_given_h(hc)
        h = self.sample_h_given_v(v)

        discarded = 0
        vhist = [v]
        vchist = []

        for i in range(max_steps):
            vc, hc, v, h, disc = self.max_coup(vc, hc, v, h, max_try=10)
            discarded += disc
            vhist.append(v)
            vchist.append(vc)
            if i >= min_steps - 1 and nd.norm(v - vc).asscalar() == 0 and nd.norm(h - hc).asscalar() == 0:
                break

        return nd.stack(*vhist), nd.stack(*vchist), discarded



class RBMucd:
    # Constructor
    def __init__(self, m, n, w0=None, b0=None, c0=None, ctx=mx.cpu()):
        self.ctx = ctx

        # Dimensions
        self.m = m
        self.n = n

        # Parameters
        if w0 is not None and b0 is not None and c0 is not None:
            if w0.shape != (m, n):
                raise ValueError("w0 must be an [m x n] array")
            if b0.shape != (m, ):
                raise ValueError("b0 must be an [m] array")
            if c0.shape != (n, ):
                raise ValueError("c0 must be an [n] array")
            self.w = w0.as_in_context(self.ctx)
            self.b = b0.as_in_context(self.ctx)
            self.c = c0.as_in_context(self.ctx)
        else:
            self.w = nd.random.normal(scale=0.1, shape=(m, n), ctx=self.ctx)
            self.b = nd.zeros(shape=m, ctx=self.ctx)
            self.c = nd.zeros(shape=n, ctx=self.ctx)
            #self.b = nd.random.normal(scale=0.1, shape=m, ctx=self.ctx)
            #self.c = nd.random.normal(scale=0.1, shape=n, ctx=self.ctx)

        # Gradients
        self.dw1 = nd.empty(shape=self.w.shape, ctx=self.ctx)
        #self.db1 = nd.empty(shape=self.b.shape, ctx=self.ctx)
        #self.dc1 = nd.empty(shape=self.c.shape, ctx=self.ctx)

        self.dw2 = nd.empty(shape=self.w.shape, ctx=self.ctx)
        #self.db2 = nd.empty(shape=self.b.shape, ctx=self.ctx)
        #self.dc2 = nd.empty(shape=self.c.shape, ctx=self.ctx)

    # Approximate log-likelihood value
    def loglik(self, dat, nobs=100, nmc=30, nstep=10):
        if nobs > dat.shape[0]:
            nobs = dat.shape[0]
        ind = np.random.choice(dat.shape[0], nobs, replace=False)
        samp = RBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        loglik = 0.0
        for i in range(nobs):
            vi = dat[ind[i]]
            v, h = samp.sample_k(vi.reshape(1, -1).repeat(nmc, axis=0), nstep)
            vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
            logp = nd.log(vmean) * vi + nd.log(1 - vmean) * (1 - vi)
            logp = nd.sum(logp, axis=1)
            loglik += log_sum_exp(logp)
        return loglik - nobs * math.log(nmc)

    # Exact log-likelihood value
    # dat[N x m]
    def loglik_exact(self, dat):
        N = dat.shape[0]

        # log(Z)
        # https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
        vperm = nd.array(vec_bin_array(np.arange(2 ** self.m), self.m), ctx=self.ctx)
        vpermwc = nd.dot(vperm, self.w) + self.c
        logzv = nd.dot(vperm, self.b) + nd.sum(nd.log(1 + nd.exp(vpermwc)), axis=1)
        logz = log_sum_exp(logzv)

        # https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
        term1 = nd.dot(dat, self.b)
        term2 = nd.log(1 + nd.exp(nd.dot(dat, self.w) + self.c))
        loglik = nd.sum(term1) + nd.sum(term2) - logz * N
        return loglik.asscalar()

    # First term of the gradient
    # Mini-batch vmb [b x m]
    def compute_grad1(self, vmb):
        # Mean of hidden units given vmb
        hmean = nd.sigmoid(nd.dot(vmb, self.w) + self.c)

        #self.db1 = nd.mean(vmb, axis=0, out=self.db1)
        #self.dc1 = nd.mean(hmean, axis=0, out=self.dc1)
        self.dw1 = nd.dot(vmb.T, hmean, out=self.dw1)
        self.dw1 /= vmb.shape[0]

    # Zero out gradients
    def zero_grad2(self):
        #self.db2 = nd.zeros_like(self.db2, out=self.db2)
        #self.dc2 = nd.zeros_like(self.dc2, out=self.dc2)
        self.dw2 = nd.zeros_like(self.dw2, out=self.dw2)

    # Compute the second term of gradient using CD-k
    # dat [N x m]
    def accumulate_grad2_cdk(self, dat, k=1, nchain=1):
        # Initial values for Gibbs sampling
        N = dat.shape[0]
        ind = np.random.choice(N, nchain)
        v0 = dat[ind, :]

        # Gibbs samples
        samp = RBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        v, h = samp.sample_k(v0, k=k)

        # Second term
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        #self.db2 = nd.sum(v, axis=0, out=self.db2)
        #self.dc2 = nd.sum(hmean, axis=0, out=self.dc2)
        self.dw2 = nd.dot(v.T, hmean, out=self.dw2)

    # Compute the second term of gradient using unbiased CD
    # dat [N x m]
    def accumulate_grad2_ucd(self, dat, min_mcmc=1, max_mcmc=100):
        # Initial value for Gibbs sampling
        N = dat.shape[0]
        ind = np.random.choice(N, 1)[0]
        v0 = dat[ind, :]

        # Gibbs samples
        samp = UnbiasedRBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        vhist, vchist, disc = samp.sample(v0, min_steps=min_mcmc, max_steps=max_mcmc)

        burnin = min_mcmc - 1
        tau = vchist.shape[0]
        remain = tau - burnin

        vk = vhist[burnin, :]
        hk_mean = nd.sigmoid(nd.dot(self.w.T, vk) + self.c)

        hhist_mean = nd.sigmoid(nd.dot(vhist[-remain:, :], self.w) + self.c)
        hchist_mean = nd.sigmoid(nd.dot(vchist[-remain:, :], self.w) + self.c)

        # Second term
        #self.db2 += vk + nd.sum(vhist[-remain:, :], axis=0) -\
        #            nd.sum(vchist[-remain:, :], axis=0)
        #self.dc2 += hk_mean + nd.sum(hhist_mean, axis=0) -\
        #            nd.sum(hchist_mean, axis=0)
        self.dw2 += nd.dot(vk.reshape(-1, 1), hk_mean.reshape(1, -1)) +\
                    nd.dot(vhist[-remain:, :].T, hhist_mean) -\
                    nd.dot(vchist[-remain:, :].T, hchist_mean)

        return tau, disc

    # Update parameters
    def update_param(self, lr, nchain):
        #self.b += lr * (self.db1 - self.db2 / nchain)
        #self.c += lr * (self.dc1 - self.dc2 / nchain)
        self.w += lr * (self.dw1 - self.dw2 / nchain)

    # Train RBM using CD-k
    def train_cdk(self, dat, batch_size, epochs, W_true, lr=0.01, k=1, nchain=1, report_freq=1, exact_loglik=False):
        N = dat.shape[0]
        ind = np.arange(N)
        loglik = []
        distW = []
        #distB = []

        for epoch in range(epochs):
            np.random.shuffle(ind)

            for i in range(0, N, batch_size):
                ib = i // batch_size + 1
                batchid = ind[i:(i + batch_size)]
                vmb = dat[batchid, :]

                self.compute_grad1(vmb)
                self.zero_grad2()
                self.accumulate_grad2_cdk(dat, k, nchain)
                self.update_param(lr, nchain)

                if ib % report_freq == 0:
                    if exact_loglik:
                        ll = self.loglik_exact(dat)
                    else:
                        ll = self.loglik(dat, nobs=100)
                    loglik.append(ll)
                    distw = nd.norm(self.w-W_true).asscalar()/(self.n*self.m)
                    distW.append(distw)
                    #distb = nd.norm(self.b).asscalar()/self.m + nd.norm(self.c).asscalar()/self.n
                    #distB.append(distb)
                    print("epoch = {}, batch = {}, loglik = {}, distW = {}".format(epoch, ib, ll, distw))

        return loglik, distW#, distB

    # Train RBM using Unbiased CD
    def train_ucd(self, dat, batch_size, epochs, W_true, lr=0.01, min_mcmc=1, max_mcmc=100, nchain=1, report_freq=1, exact_loglik=False):
        N = dat.shape[0]
        ind = np.arange(N)
        loglik = []
        tau = []
        disc = []
        distW = []
        #distB = []

        for epoch in range(epochs):
            np.random.shuffle(ind)

            tt = 0.0
            dd = 0.0
            for i in range(0, N, batch_size):
                ib = i // batch_size + 1
                batchid = ind[i:(i + batch_size)]
                bs = batchid.size
                vmb = dat[batchid, :]

                self.compute_grad1(vmb)
                self.zero_grad2()
                for j in range(nchain):
                    tau_t, disc_t = self.accumulate_grad2_ucd(dat, min_mcmc=min_mcmc, max_mcmc=max_mcmc)
                    tt += tau_t
                    dd += disc_t
                self.update_param(lr, nchain)

                if ib % report_freq == 0:
                    if exact_loglik:
                        ll = self.loglik_exact(dat)
                    else:
                        ll = self.loglik(dat, nobs=100)
                    loglik.append(ll)
                    tau.append(tt / nchain / report_freq)
                    disc.append(dd / nchain / report_freq)
                    distw = nd.norm(self.w - W_true).asscalar()/(self.n*self.m)
                    distW.append(distw)
                    #distb = nd.norm(self.b).asscalar()/self.m + nd.norm(self.c).asscalar()/self.n
                    #distB.append(distb)
                    tt = 0.0
                    dd = 0.0
                    print("epoch = {}, batch = {}, loglik = {}, distW = {}".format(epoch, ib, ll, distw))

        return loglik, tau, disc, distW#, distB


