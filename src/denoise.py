import pywt
import math
import numpy as np
from scipy import stats

class DeNoise:
    def __init__(self, segment):
        self.df = segment.copy()
        self.df.set_index('flow', inplace = True)

    def load_flow(self, flow, field = 'owd'):
        # df1 = self.df
        # df1 = df1[df1.flow == flow]
        # self.flow = df1[field]
        self.flow = self.df.loc[flow, field]
        self.n = len(self.flow)

    def dwt(self, level, wavelet = 'db1'):
        """level: int, the decomposition level (must be >= 0) in discrete wavelet transform."""
        coeffs = pywt.wavedec(self.flow, wavelet, level = level)
        return coeffs # list of [cA_n, cD_n, cD_n-1, …, cD2, cD1]

    def threshold(self, coeffs):
        """compute the threshold value t"""
        coe_list = []
        for coeff in coeffs:
            temp = list(coeff)
            coe_list.extend(temp)
        coe_array = np.array(coe_list)
        # noise estimate = MAD / 0.6745 with with MAD the median absolute value 
        # of the appropriately normalized fine-scale wavelet coefficients
        noise = stats.median_abs_deviation(coe_array * (self.n)**0.5) / 0.6745
        # threshold =  noise * (log (n)/n) ** 0.5
        self.t = noise * (2 * math.log(self.n) / self.n) ** 0.5
        if self.t == 0:
            self.t = 0.00001 # avoid 0 divid
        return self.t

    def soft_threshold(self, coeffs):
        """return the new coeffcients after soft thresholding"""
        new_coeffs = []
        for ci in coeffs:
            ci = pywt.threshold(ci, self.t, 'soft')
            new_coeffs.append(ci)
        return new_coeffs

    def idwt(self, coeffs):
        # coeffs is a list of [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        # level here is n
        result = pywt.waverec(coeffs, 'db1') 
        return result