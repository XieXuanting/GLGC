#!usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import time
import random
# import tensorflow as tf
import numpy as np
import scipy
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from time import *
import warnings
import core_sampling
warnings.filterwarnings("ignore")
# warnings.simplefilter('error',ComplexWarning)


def read_data(str):
    data = sio.loadmat('./Data/{}.mat'.format(str))
    if(str == 'large_cora'):
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else:
        X = data['fea']
        A = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]

    return X, A, gnd


def FGC_cora_modified(X, A, gnd, a, k, ind):
    # Store some variables

    N = X.shape[0]
    # print("N = {}".format(N))
    Im = np.eye(len(ind))
    In = np.eye(N)
    if sp.issparse(X):
        X = X.todense()

    # Normalize A
    A = A + In
    D = np.sum(A, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)

    # Get filter G
    Ls = In - A
    G = In - 0.5 * Ls
    G_ = In
    X_hat = X
    for i in range(k):
        X_hat = G.dot(X_hat)
    begin_time = time()


    # you can modify the order of f(A) here
    # default: f(A) = A

    A_hat = A[ind]  # (m,n)
    B_hat = X_hat[ind]  # (m,d)
    # Set the order of filter
    tmp1 = (B_hat.dot(B_hat.T) + a * Im)
    tmp2 = (B_hat.dot(X_hat.T) + a * A_hat)
    S = np.linalg.inv(tmp1).dot(tmp2)

    return S, begin_time


def main(X, A, gnd, m, a, k, ind):

    N = X.shape[0]
    begin_time_filter = time()
    types = len(np.unique(gnd))
    S, begin_time = FGC_cora_modified(X, A, gnd, a, k, ind)
    D = np.sum(S, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)

    S_hat = D.dot(S)

    S_hat_tmp = S_hat.dot(S_hat.T)  # (m,m)
    S_hat_tmp[np.isinf(S_hat_tmp)] = 0
    S_hat_tmp[np.isnan(S_hat_tmp)] = 0
    # sigma, E = scipy.linalg.eig(S_hat_tmp)
    E, sigma, v = sp.linalg.svds(S_hat_tmp, k=types, which='LM')
    sigma = sigma.T
    sigma = np.power(sigma, -0.5)
    sigma[np.isinf(sigma)] = 0
    sigma[np.isnan(sigma)] = 0
    sigma = np.diagflat(sigma)
    C_hat = (sigma.dot(E.T)).dot(S_hat)
    C_hat[np.isinf(C_hat)] = 0
    C_hat[np.isnan(C_hat)] = 0
    C_hat = C_hat.astype(float)
    kmeans = KMeans(n_clusters=types, random_state=37).fit(C_hat.T)

    predict_labels = kmeans.predict(C_hat.T)

    cm = clustering_metrics(gnd, predict_labels)
    ac, nm, f1 = cm.evaluationClusterModelFromLabel()
    end_time = time()
    tot_time = end_time - begin_time
    tot_time_filter = end_time - begin_time_filter
    return ac, nm, f1, tot_time, tot_time_filter


def lower_bound(p, rd):
    l = 0
    r = len(p) - 1
    while(l < r):
        mid = (l + r) // 2
        if(p[mid] > rd):
            r = mid
        else:
            l = mid + 1
    # print("rd = {}, l = {}, r= {}".format(rd, l, r))
    return l


def node_sampling(A, m, alpha):
    D = np.sum(A, axis=1).flatten()
    if(len(np.shape(D)) > 1):
        D = D.A[0]
    D = D**alpha
    tot = np.sum(D)
    p = D / tot

    for i in range(len(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    ind = []
    vis = [0] * len(D)
    while(m):
        while(1):
            rd = np.random.rand()
            pos = lower_bound(p, rd)
            if(vis[pos] == 1):
                continue
            else:
                vis[pos] = 1
                ind.append(pos)
                m = m - 1
                break
    return ind


def func(X, A, gnd):
    # numbers of anchor
    i = 40

    # setting other parameters
    a = 1
    k = 10
    alpha = 1
    k_list = []
    aa_list = []
    i_list = []
    ac_list = []
    nm_list = []
    f1_list = []
    tm_list = []
    tm_list_filter = []
    f_alpha_list = []

    N = X.shape[0]
    tot_test = 1
    ac_max = 0.0
    xia = 0
    tot = 0



    # core sampling
    distb = core_sampling.get_distribution('core', alpha, A)
    ind, ___, __ = core_sampling.node_sampling(A, distb, i)
    ac_mean = 0
    nm_mean = 0
    f1_mean = 0
    tm_mean = 0

    # continue
    acc, nmm, f11, tm, tm_filter = main(
        X, A, gnd, i, a, k, ind)
    print("m = {},k = {}, f_alpha = {},a  ={}, ac = {}, nmi = {}, f1 = {}, tm = {}, tm_filter = {}".format(
        i, k, alpha, a, acc, nmm, f11, tm, tm_filter))
    if(ac_mean < acc):
        ac_mean = acc
        nm_mean = nmm
        f1_mean = f11
        tm_mean = tm
        tm_mean_filter = tm_filter
    i_list.append(i)
    k_list.append(k)
    aa_list.append(a)
    f_alpha_list.append(alpha)
    ac_list.append(ac_mean)
    nm_list.append(nm_mean)
    f1_list.append(f1_mean)
    tm_list.append(tm_mean)
    tm_list_filter.append(tm_mean_filter)
    print("m = {}, k ={},f_alpha = {}, ac_mean = {},nm_mean = {},f1_mean = {},tm_mean = {},tm_mean_filter = {}\n".format(
        i, k, alpha, ac_mean, nm_mean, f1_mean, tm_mean, tm_mean_filter))

    if(ac_mean > ac_max):
        xia = tot
        ac_max = ac_mean

    tot += 1


    print("the result is ")
    print("m = {},k = {},f_alpha = {}, ac_mean = {}, nm_mean = {}, f1_mean = {},tm_mean = {},tm_mean_filter = {}".format(
        i_list[xia], k_list[xia], f_alpha_list[xia], ac_list[xia], nm_list[xia], f1_list[xia], tm_list[xia], tm_list_filter[xia]))
    return i_list[xia], k_list[xia], f_alpha_list[xia], ac_list[xia], nm_list[xia], f1_list[xia], tm_list[xia], tm_list_filter[xia]


if __name__ == '__main__':
    dataset = 'cora'
    X, A, gnd = read_data(dataset)
    # number of epoch
    tt = 1
    m_best_list = []
    k_best_list = []
    f_alpha_best_list = []
    ac_best_list = []
    nm_best_list = []
    f1_best_list = []
    tm_best_list = []
    tm_filter_best_list = []
    for i in range(tt):
        nowm, nowk, nowf, nowac, nownm, nowf1, nowtm, nowtmf = func(X, A, gnd)
        m_best_list.append(nowm)
        k_best_list.append(nowk)
        f_alpha_best_list.append(nowf)
        ac_best_list.append(nowac)
        nm_best_list.append(nownm)
        f1_best_list.append(nowf1)
        tm_best_list.append(nowtm)
        tm_filter_best_list.append(nowtmf)

