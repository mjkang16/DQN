
import numpy as np
import tensorflow as tf

def epsilonmax(x, eps):
    n = len(x)
    p = np.ones(n)*eps/n
    p[np.argmax(x)] += 1 - eps
    
    return p

def softmax(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    p = e_x/e_x.sum()
    p = p/p.sum()
    return p

def softV(z, scale = 1.):
    z = np.array(z)/scale
    max_z = np.max(z, axis=1)
    e_z = np.exp(z - max_z[:, np.newaxis])
    e_sum = np.sum(e_z, axis=1)
    e_sum = scale * (np.log(e_sum) + max_z)
    return e_sum
"""
def sparsetau(x):
    x = np.array(x)
    sorted_x = np.sort(x)[::-1]
    S = np.array([])
    for i in range(0,len(x)):
        if 1+(i+1)*sorted_x[i] >= (sorted_x[0:(i+1)]).sum():
            S = np.append(S,sorted_x[i])
    tau = (S.sum() - 1)/S.size
    return tau, S
"""
def sparsedist(z, scale=1.):
    z = np.array(z/scale)
    if len(z.shape) == 1:
        z = np.reshape(z,(1,-1))
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    p = np.maximum(0, z - tau_z)
    
    return p

def sparsemax(z, scale=1.):
    z = np.array(z/scale)
    x = np.mean(z, axis=1)
    z = z - x[:, np.newaxis]
    
    # calculate sum over S(z)
    p = sparsedist(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    return scale*(0.5 * S_sum + 0.5 + x)
                 
"""
def sparse_eps_dist(x, eps, scale = 1):
    x = np.array(x/scale)
    tau, _ = sparsetau(x)
    p = x - tau
    p[p<0] = 0
    if p.sum() > 0.0:
        p = p/p.sum()
    else:
        p = np.ones_like(x)/x.shape[0]
    
    n = len(x)
    p = np.ones(n) * eps/n + p*(1.-eps)
    return p
"""
"""
def sparsemax(A_batch,scale = 1):
    Q_batch = []
    for x in A_batch:
        x = np.array(x/scale)
        tau, S = sparsetau(x)
        spmax_x = 0.5*(S**2 - tau**2).sum() + 0.5
        Q_batch.append(scale*spmax_x)
    return Q_batch
"""

def choice_action(Exp, eps, scale, action_Q):
    if Exp == 'softmax':
        action_max = softmax(action_Q, scale)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    elif Exp == 'sparsemax':
        action_dist = sparsedist(action_Q, scale)[0]
        n = len(action_dist)
        action_dist = np.ones(n) * eps/n + action_dist*(1.-eps)
        action_dist = action_dist/np.sum(action_dist)
        return np.random.choice(len(action_dist),size=1,p=action_dist)[0]
    elif Exp == 'epsilon':
        action_max = epsilonmax(action_Q, eps)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    else:
        print("Exploration Type is wrong.")
        return 0
    
        