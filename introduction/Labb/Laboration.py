import numpy as np
import scipy.stats as stats


class LinearRegression:
   
    def __init__(self, Y, X):
        self.Y = Y
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.n = self.X.shape[0]
        self.d = self.X.shape[1] - 1
        self.b = self.fit()
        self.SSE = np.sum(np.square(self.Y-(self.X @ self.b)))
        self.Syy = (self.n *np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
        self.variance = self.calc_variance()
        self.SSR = self.Syy - self.SSE
        self.S = np.sqrt(self.variance)
        
  
    
    def fit(self):
        self.b = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        return self.b


    def calc_variance(self):
        variance = self.SSE / (self.n - self.d -1)
        return variance
        


    def calc_std(self):
       standard_deviation = self.variance ** 0.5
       return standard_deviation


    def significance_regression(self):
        sig_statistic = (self.SSR/self.d)/self.S
        p_significance = stats.f.sf(sig_statistic, self.d, self.n-self.d-1)

        return p_significance
        
        


    def relevance_regression(self):
       Rsq = self.SSR / self.Syy
       return Rsq
    



    def individual_significance(self): 
        c = np.linalg.pinv(self.X.T @ self.X)*self.variance
        p_b3_list = []

        for j in range(len(self.b)):
            if j == 0:
                continue
            b3_statistic = self.b[j] / (self.S*np.sqrt(c[j,j]))
        
            p_b3 = 2*min(stats.t.cdf(b3_statistic, self.n-self.d-1), stats.t.sf(b3_statistic, self.n-self.d-1))
            p_b3_list.append(p_b3)
            
        return p_b3_list
        


    def Pearson(self):
        pearson_list = []
        
        for i in range(self.d):
            for j in range(i + 1, self.d):
                pearson = stats.pearsonr(self.X[:, i], self.X[:, j])
                pearson_list.append((i + 1, j + 1, pearson))

    
        return pearson_list
   


    def confidence_intervals(self): # for loop
        Sxx = np.sum(np.square(self.X)) - (np.square(np.sum(self.X)) / self.n)
        se_b = self.variance/Sxx
        ci = (self.b[1], 2*np.sqrt(se_b))
        return ci

    
    
    def confidence_level(self):
        pass
    # last add a property confidence level that stores the selected confidence level.
