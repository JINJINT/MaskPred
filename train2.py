from RBM import *
from utils import *
from RBMbase import *
from RBMucd import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns   

if not os.path.exists('./plot'):
    os.makedirs('./plot')


mx.random.seed(123)
random.seed(123)
ctx = mx.cpu()

n_train = 1000
n_valid = 1000
n_visible = 25 # v
n_hidden = 9  # h
n_batch = 100
lr = 0.1

w = nd.random.normal(shape=(n_visible, n_hidden), ctx=ctx)
b = nd.random.normal(shape=n_visible, ctx=ctx)
c = nd.random.normal(shape=n_hidden, ctx=ctx)

# # Test CD-k sampler
# sampler0 = RBMSampler(w, b, c, ctx=ctx)
# nchain = 3
# v0 = nd.random.randint(0, 2, shape=(nchain, m), ctx=ctx).astype(w.dtype)
# res = sampler0.sample_k(v0, k=100)
# print(res)

# # Test unbiased MCMC sampler
# mx.random.seed(123)
# sampler = UnbiasedRBMSampler(w, b, c, ctx=ctx)
# v0 = nd.random.randint(0, 2, shape=m, ctx=ctx).astype(w.dtype)
# res = sampler.sample(v0, min_steps=1, max_steps=100)
# print(res)

W_truemat = generate_W('random', n_hidden, n_visible, p=0.1)
W_true = torch.tensor(W_truemat, dtype=torch.float32)
n_hidden, n_visible = W_true.shape

# Generate data from W_true using classical Gibbs
# rbm_true =  RBM(n_visible, n_hidden, n_train + n_valid + 1000, lr, n_batch, seed=123, W_init = torch.clone(W_true))
# samples = rbm_true.samp(n_train + 1000)
# samples_converged = samples[1001:,]
# shuffle_indices = np.arange(0, n_train + n_valid)
# np.random.shuffle(shuffle_indices)
# samples_train = samples_converged[shuffle_indices[:n_train],]
# samples_valid = samples_converged[shuffle_indices[n_train:],]
# print(samples_train.shape)
# print(samples_valid.shape)

# Generate data from W_true using unbiased Gibbs 
mx.random.seed(123)
b = nd.zeros(shape=n_visible, ctx=ctx)
c = nd.zeros(shape=n_hidden, ctx=ctx)
w = nd.array(np.transpose(W_truemat),ctx=ctx)
sampler = UnbiasedRBMSampler(w, b, c, ctx=ctx)
v0 = nd.random.randint(0, 2, shape=n_visible, ctx=ctx).astype(w.dtype)
samples_train = nd.zeros(shape=(n_train, n_visible), ctx=ctx)
for i in range(n_train):
    samples_train[i,]=sampler.sample(v0, min_steps=1, max_steps=1000)[0][0,]



ctx = mx.cpu()
dat = nd.array(samples_train, ctx=ctx)

# Train RBM using CD-k
cd1 = RBMucd(n_visible, n_hidden, ctx=ctx)
res_cd1 = cd1.train_cdk(dat, W_true = nd.array(np.transpose(W_truemat),ctx=ctx), 
                        batch_size=n_batch, epochs=200, lr=lr,
                        k=1, nchain=1000,
                        report_freq=1, exact_loglik=False)


# Train RBM using UCD
ucd = RBMucd(n_visible, n_hidden, ctx=ctx)
res_ucd = ucd.train_ucd(dat, W_true = nd.array(np.transpose(W_truemat),ctx=ctx), 
                        batch_size=n_batch, epochs=200, lr=lr,
                        min_mcmc=1, max_mcmc=100, nchain=1000,
                        report_freq=1, exact_loglik=False)


# plot results

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
n = len(res_cd1[0])
ax1.plot(np.arange(n), res_ucd[0], label = 'UCD')
ax1.plot(np.arange(n), res_cd1[0], label = 'CDk')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Log-likelihood')
ax1.set_title('Log-likelihood')
ax1.legend()

ax2.plot(np.arange(n), res_ucd[3], label = 'UCD')
ax2.plot(np.arange(n), res_cd1[1], label = 'CDk')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Weight differences')
ax2.set_title('weight diff')
ax2.legend()

ax3.plot(np.arange(n), res_ucd[4], label = 'UCD')
ax3.plot(np.arange(n), res_cd1[2], label = 'CDk')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Bias differences')
ax3.set_title('bias diff')
ax3.legend()



fig.tight_layout()
method= 'random'
max_epoch = n

plt.savefig("./plot/new1000ucdRBM_W{}_h{}_v{}_lr{}_batch{}_epoch{}.png".format(method,n_hidden,n_visible,lr,n_batch,max_epoch), format="png")


# fig = plt.figure()
# sub = fig.add_subplot(131)
# n = len(res_cd1[0])
# sns.lineplot(np.arange(n), res_ucd[0], label="Unbiased CD")
# sns.lineplot(np.arange(n), res_cd1[0], label="CD")
# sub.set_xlabel("Iteration")
# sub.set_ylabel("Log-likelihood Function Value")

# sub = fig.add_subplot(132)
# sns.lineplot(np.arange(n), res_ucd[3], label="Unbiased CD")
# sns.lineplot(np.arange(n), res_cd1[1], label="CD")
# sub.set_xlabel("Iteration")
# sub.set_ylabel("Weight differences")

# sub = fig.add_subplot(133)
# sns.lineplot(np.arange(n), res_ucd[4], label="Unbiased CD")
# sns.lineplot(np.arange(n), res_cd1[2], label="CD")
# sub.set_xlabel("Iteration")
# sub.set_ylabel("Bias differences")

# sub = fig.add_subplot(132)
# sns.lineplot(np.arange(n), res_ucd[1])
# sub.set_xlabel("Iteration")
# sub.set_ylabel("Average Stopping Time")

# sub = fig.add_subplot(133)
# sns.lineplot(np.arange(n), res_ucd[2])
# sub.set_xlabel("Iteration")
# sub.set_ylabel("# Rejected Samples per Chain")

# fig.show()






