import numpy as np
import scipy.stats as stats


class LinearRegression:
   
    def __init__(self, Y, X):
        self.Y = Y
        self.X_name = X
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.b = self.fit()
        self.SSE = np.sum(np.square(self.Y-(self.X @ self.b)))
        self.Syy = (self.n *np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
        self.variance = self.calc_variance()
        self.std = self.calc_std()
        self.SSR = self.Syy - self.SSE
        self.S = np.sqrt(self.variance)
        
    @property
    def column_names(self):
        return list(self.X_name.columns)

    
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
        sig_statistic = (self.SSR/self.d)/self.variance
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
        XTX_inv = np.linalg.pinv(self.X.T @ self.X) *self.variance  # Inverse of X'X
        alpha = 1 - self.confidence_level 
        t_critical = stats.t.ppf(alpha / 2, self.n - self.d - 1)
        

        ci = []
        for i in range(1,len(self.b)):
            low = self.b[i] - t_critical *self.std * np.sqrt(XTX_inv[i][i])
            high = self.b[i] + t_critical *self.std * np.sqrt(XTX_inv[i][i])  
            ci.append((self.column_names[i-1], low, high)) 

        return ci

    
    @property
    def confidence_level(self):
        R2 = self.relevance_regression()
        if R2 >= 0.997:
            confidence_level = 0.997
            return confidence_level
        elif R2 >= 0.95:
            confidence_level = 0.95
            return confidence_level
        elif R2 >= 0.68:
            confidence_level = 0.68
            return confidence_level
        else: 
            print("The relevance of your model is too low!")
            
    

    def observer_bias(self, feature):
        if feature not in self.column_names:
            raise ValueError(f"Feature: {feature} not found in the model.")
        
        self.column_names.index(feature)
        j = self.column_names.index(feature) + 1
        
        c = np.linalg.pinv(self.X.T @ self.X)*self.variance
        b3_statistic = self.b[j] / (self.S*np.sqrt(c[j,j]))
        p_b3 = 2*min(stats.t.cdf(b3_statistic, self.n-self.d-1), stats.t.sf(b3_statistic, self.n-self.d-1))
        
            
        return feature, p_b3 



