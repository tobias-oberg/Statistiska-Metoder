import numpy as np
import scipy.stats as stats

# Syy = (n*np.sum(np.square(Y)) - np.square(np.sum(Y)))/n
# SSR = Syy - SSE

class LinearRegression:
   
    def __init__(self, Y, X):
        self.Y = Y
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.n = self.X.shape[0]
        self.k = self.X.shape[1] - 1
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

        variance = np.var(self.Y)
        return variance
        


    def calc_std(self):
       standard_deviation = np.std(self.Y)
       return standard_deviation


    def significance_regression(self):
        n = len(self.X)
        k = len(self.b)-1
        
        
        

        sig_statistic = (self.SSR/self.k)/self.S
        p_significance = stats.f.sf(sig_statistic, k, n-k-1)

        return p_significance
        
        # finding p-value, SSR = Syy - SSE



    def relevance_regression(self):
        c = np.linalg.pinv(self.X.T @ self.X)*self.variance
        b3_statistic = self.b[3] / (self.S*np.sqrt(c[3,3]))
        p_b3 = 2*min(stats.t.cdf(b3_statistic, self.n-self.k-1), stats.t.sf(b3_statistic, self.n-self.k-1))
        
        Rsq = self.SSR / self.Syy

        return Rsq, p_b3
    
    
    
    # def relevance_regression(self):

    #     SSE = np.sum(np.square(self.Y - (self.X @ self.b)))  # Sum of squared errors
    #     Syy = (self.n * np.sum(np.square(self.Y)) - np.square(np.sum(self.Y))) / self.n  # Total sum of squares
    #     SSR = Syy - SSE  # Regression sum of squares
    #     Rsq = SSR / Syy  # R-squared

    #     # Calculate coefficient p-values
    #     MSE = SSE / (self.n - self.k - 1)  # Mean square error
    #     cov_matrix = np.linalg.pinv(self.X.T @ self.X) * MSE  # Covariance matrix of coefficients
    #     t_stats = self.b / np.sqrt(np.diag(cov_matrix))  # t-statistics for each coefficient
    #     p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=self.n - self.k - 1)) for t in t_stats]  # p-values

    #     return Rsq, t_stats, p_values
    
    











# if __name__ == "__main__":
#     main()