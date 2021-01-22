from numba.core.decorators import njit
import numpy as np
import numba as nb
import abc
from numba import vectorize, float64,int64

class Bases(metaclass=abc.ABCMeta):

#     # @staticmethod
#     # def heaviside():
#     #     return fb.fast_steps_heaviside()
        
#     # @staticmethod
#     # def heaviside_old(x,s, v, d):
#     #     return np.where(x >= 0,s,0)

#     # @staticmethod
#     # def constant(x,s, v, d):
#     #     return np.ones(len(x))
    
#     # @staticmethod
#     # def logit_old(x,s, v, d):
#     #     return 0.5*(1+np.tanh(x/s))

#     # @staticmethod
#     # def logit():
#     #     return fb.fast_steps_logit
    
#     # @staticmethod
#     # def expon(x,s, v, d):
#     #     return (1 - np.exp(-s*np.sign(x)))
    
#     # @staticmethod
#     # def arctan(x,s, v, d):
#     #     return (0.5+(1/np.pi)*np.arctan(x/s))
    
#     # @staticmethod
#     # def sigmoid(x,s, v, d):
#     #     return 1.0/(1.0+np.exp(-s*x))
    
#     # @staticmethod
#     # def norm(x,s, v, d):
#     #     k = s*np.sqrt(2*np.pi)
#     #     return np.exp(-0.5*(x/s)**2)/k

#     # @staticmethod
#     # def sinc(x,s, v, d):
#     #     return np.sinc(x*s)

    @staticmethod
    @vectorize([float64(float64,float64, float64,float64)],nopython=True, target='parallel')
    def fconstant(x,step,direction,weight):
        return weight

    @staticmethod
    @vectorize([float64(float64,float64, float64,float64),int64(int64,int64, int64,int64)],nopython=True, target='parallel')
    def fheaviside(x,step,direction,weight):
        return weight if direction*(x-step)>=0 else 0

    @staticmethod
    @vectorize([float64(float64,float64),int64(int64,int64)],nopython=True)
    def fcumsum(a,b):
        return a+b

    @staticmethod
    @njit(parallel=True,nogil=True)
    def ffheaviside(x,steps,param):
        result = np.zeros(len(x))
        for i in nb.prange(steps.shape[0]):
            result += np.where(steps[i,1]*(x-steps[i,0])>=0, steps[i,2],0)

        return result

    @staticmethod
    @njit(parallel=True,nogil=True)
    def fflogit(x,steps,param):
        result = np.zeros(len(x))
        for i in nb.prange(steps.shape[0]):
            result += steps[i,2]*0.5*(1+np.tanh(steps[i,1]*(x-steps[i,0])/param))

        return result

    @staticmethod
    @njit(parallel=True,nogil=True)
    def fsearchsort(values, keys, x):
        return values[np.searchsorted(keys,x,side = 'left')]