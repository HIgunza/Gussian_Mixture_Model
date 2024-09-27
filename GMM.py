import numpy as np
from scipy import random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, c, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None):
        '''
            - c: Number of Gaussian clusters
            - dim: Dimension 
            - mu: initial value of mean of clusters (c, dim)
                        random from uniform[-10, 10]
            - sigma: initial value of covariance matrix of clusters (c, dim, dim)
                           Identity matrix for each cluster
            - pi: initial value of cluster weights (c,)
                        equal value to all cluster i.e. 1/c
            - colors: Color valu for plotting each cluster (c, 3)
                       random from uniform[0, 1]
        '''
        self.c = c
        self.dim = dim
        if(init_mu is None):
            init_mu = random.rand(c, dim)*20 - 10
        self.mu = init_mu
        if(init_sigma is None):
            init_sigma = np.zeros((c, dim, dim))
            for i in range(c):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if(init_pi is None):
            init_pi = np.ones(self.c)/self.c
        self.pi = init_pi
        if(colors is None):
            colors = random.rand(c, 3)
        self.colors = colors
    
    def init_em(self, D):
        '''
        Initialization for EM algorithm.
        input: D: data 
        '''
        self.data = D
        self.num_points = D.shape[0]
        self.z = np.zeros((self.num_points, self.c))
    
    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.c):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.c):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
            
    def log_likelihood(self, D):
        '''
        Compute the log-likelihood of D under current parameters
        input: D: Data 
        output:log-likelihood of D: Sum_n Sum_k log(pi_k * N( D_n | mu_k, sigma_k ))
        '''
        logli = []
        for d in D:
            tot = 0
            for i in range(self.c):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            logli.append(np.log(tot))
        return np.sum(logli)
    
    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
         function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
       
        '''
        if(self.dim != 2):
            print("Drawing for 2D case.")
            return
        for i in range(self.c):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)