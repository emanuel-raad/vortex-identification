import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from visualize import visualize_tensor
from filters.vectorMedianFilter import vectorMedianFilter
from filters.anpa import anpaFilterSpatial, anpaFilterTemporal
from filters.exponential import exponentialSmoothing
from decompose import velocity_decomposition
from criteria import q_criterion
from threshold import localDiffStencil

MEDIUM_SIZE = 22

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['legend.fontsize'] = MEDIUM_SIZE

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

timeseries_x = np.load(os.path.join('data', 'processed', 'timeseries_x.npy'))
timeseries_y = np.load(os.path.join('data', 'processed', 'timeseries_y.npy'))

timeseries_u = np.load(os.path.join('data', 'processed', 'timeseries_u_noisy.npy'))
timeseries_v = np.load(os.path.join('data', 'processed', 'timeseries_v_noisy.npy'))

timeseries_uf = np.load(os.path.join('data', 'processed', 'timeseries_uf.npy'))
timeseries_vf = np.load(os.path.join('data', 'processed', 'timeseries_vf.npy'))

timeseries_uff_s = np.load(os.path.join('data', 'processed', 'timeseries_uff_s.npy'))
timeseries_vff_s = np.load(os.path.join('data', 'processed', 'timeseries_vff_s.npy'))

timeseries_uff_t_anpa = np.load(os.path.join('data', 'processed', 'timeseries_uff_t_anpa.npy'))
timeseries_vff_t_anpa = np.load(os.path.join('data', 'processed', 'timeseries_vff_t_anpa.npy'))
timeseries_uff_t_exp = np.load(os.path.join('data', 'processed', 'timeseries_uff_t_exp_alpha_0pt8.npy'))
timeseries_vff_t_exp = np.load(os.path.join('data', 'processed', 'timeseries_vff_t_exp_alpha_0pt8.npy'))

base = os.path.join('results', 'hclust-ward')

print('Loaded')

n, m, t = np.shape(timeseries_x)
delX = timeseries_x[1,0,0] - timeseries_x[0,0,0]
delY = timeseries_y[0,1,0] - timeseries_y[0,0,0]

print('delX ', delX)
print('delY ', delY)

# --------------------------------------------------------------------------

cases = [
    'no-filter',
    'spatial-only',
    'spatial-temporal-anpa',
    'spatial-temporal-exp',
]

for c, case in enumerate(cases):

    print(case)

    if case == 'no-filter':
        timeseries_uff = timeseries_u
        timeseries_vff = timeseries_v
    elif case == 'spatial-only':
        timeseries_uff = timeseries_uff_s
        timeseries_vff = timeseries_vff_s
    elif case == 'spatial-temporal-anpa':
        timeseries_uff = 0.5 * (timeseries_uff_s + timeseries_uff_t_anpa)
        timeseries_vff = 0.5 * (timeseries_vff_s + timeseries_vff_t_anpa)
    elif case == 'spatial-temporal-exp':
        timeseries_uff = 0.5 * (timeseries_uff_s + timeseries_uff_t_exp)
        timeseries_vff = 0.5 * (timeseries_vff_s + timeseries_vff_t_exp)

    # --------------------------------------------------------------------------

    # Evaluate at one timestep

    i = 10

    fig, axes = visualize_tensor(
        [
            [ timeseries_u[:,:,i],   timeseries_v[:,:,i]   ],
            # [ timeseries_uf[:,:,i],  timeseries_vf[:,:,i]  ],
            [ timeseries_uff[:,:,i], timeseries_vff[:,:,i] ],
        ], timeseries_x[:,:,i], timeseries_y[:,:,i]
    )

    plt.savefig(os.path.join(base, str(c) + '_' + case + '_input.png'))
    # plt.show()

    # --------------------------------------------------------------------------

    L, strain, vorticity = velocity_decomposition(timeseries_uff[:,:,i], timeseries_vff[:,:,i], delX, delY)
    q = q_criterion(strain, vorticity)

    lower_thresh = 0
    thresh_indices = q < lower_thresh
    q[thresh_indices] = lower_thresh

    upper_thresh = 100
    thresh_indices = q > upper_thresh
    q[thresh_indices] = upper_thresh

    localDiff = localDiffStencil(q)

    indices = q > 1
    x_scatter = timeseries_x[:,:,i][indices]
    y_scatter = timeseries_y[:,:,i][indices]


    visualize_tensor(
        [[q, localDiff]], timeseries_x[:,:,i], timeseries_y[:,:,i], pic_settings = {'vmin':-1000, 'vmax':1000}
    )
    plt.savefig(os.path.join(base, str(c) + '_' + case + '_q_criterion.png'))
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.scatter(x_scatter, y_scatter, s=1)
    # plt.xlim(np.min(timeseries_x[:,:,i]), np.max(timeseries_x[:,:,i]))
    # plt.ylim(np.min(timeseries_y[:,:,i]), np.max(timeseries_y[:,:,i]))
    # plt.show()

    # --------------------------------------------------------------------------

    X = np.transpose(np.vstack((x_scatter, y_scatter)))
    print(np.shape(X))

    import matplotlib
    cmap_rand = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))

    # --------------------------------------------------------------------------
    # KMEANS
    
    # kmeans = KMeans(n_clusters=16)
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)

    # plt.figure(figsize=(10, 5))
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=1)

    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.2)
    
    # --------------------------------------------------------------------------
    # WARD

    n_clusters = 16
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    ward.fit(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], c=ward.labels_, cmap=cmap_rand, s=1)

    # --------------------------------------------------------------------------
    # DBSCAN
    
    # db = DBSCAN(eps=3*delX, min_samples=20)
    # db.fit(X)    

    # n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    # n_noise_ = list(db.labels_).count(-1)
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

    # plt.figure(figsize=(10, 5))
    # noise = np.where(db.labels_ == -1)
    # clusters = np.where(db.labels_ != -1)

    # # Plot the clusters in random clusters, and the noise in black

    # plt.scatter(X[:, 0][clusters], X[:, 1][clusters], c=db.labels_[clusters], cmap=cmap_rand, s=1)
    # plt.scatter(X[:, 0][noise],    X[:, 1][noise], c='black', s=1)
    
    # --------------------------------------------------------------------------
    # LABELS AND SAVING
    
    plt.xlim(np.min(timeseries_x[:,:,i]), np.max(timeseries_x[:,:,i]))
    plt.ylim(np.min(timeseries_y[:,:,i]), np.max(timeseries_y[:,:,i]))
    plt.title(case)
    plt.xlabel('x/d')
    plt.ylabel('y/d')
    plt.title('Case #{}: {}'.format(c+1, case))
    plt.tight_layout()
    plt.savefig(os.path.join(base, str(c) + '_' + case + '_hclust.png'))

    # --------------------------------------------------------------------------
