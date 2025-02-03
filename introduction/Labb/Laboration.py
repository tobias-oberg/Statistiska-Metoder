import numpy as np
import scipy.stats as stats


class LinearRegression:
   
    def __init__(self, Y, X, column_names, confidence_level = 0.95):
        self.column_names = ["Intercept"] + column_names
        self.Y = Y
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.b = self.fit()
        self.SSE = np.sum(np.square(self.Y-(self.X @ self.b)))
        self.Syy = (self.n *np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
        self.variance = self.calc_variance()
        self.SSR = self.Syy - self.SSE
        self.S = np.sqrt(self.variance)
        self.confidence_level = confidence_level
        
        

    @property 
    def d(self):
        return self.X.shape[1] - 1



    @property
    def n(self):
        return self.X.shape[0]



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
        removing_intercept = self.X[:,1:]
        
        for i in range(self.d):
            for j in range(i +1, self.d):
                pearson, p_value = stats.pearsonr(removing_intercept[:, i], removing_intercept[:, j])
                pearson_list.append((i + 1, j + 1, pearson))

    
        return pearson_list
   


    def confidence_intervals(self): 
        XTX_inv = np.linalg.pinv(self.X.T @ self.X)  # Inverse of X'X
        se_b = np.sqrt(np.diagonal(XTX_inv) * self.variance) 
    
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, self.n - self.d - 1)

        ci = []
        for i in range(len(self.b)):
            low = self.b[i] - t_critical * se_b[i]  
            high = self.b[i] + t_critical * se_b[i]  
            ci.append((self.column_names[i], low, high)) 

        return ci

    # scipy ppf Z 'alpha'/2
    
    @property
    def confidence_level(self):
        return self._confidence_level
 
    @confidence_level.setter
    def confidence_level(self, value):
        self._confidence_level = value
