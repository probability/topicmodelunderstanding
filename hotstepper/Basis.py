from __future__ import annotations
import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

from typing import Optional, Union


class Basis():
    T = Union[int,float]
    
    __slots__ = ('lbound','ubound','_base','_ibase','_dbase','param')
    
    @staticmethod
    def heaviside(x:list(T),s:float) -> np.ufunc:
        return cp.where(x >= 0,s,0)

    # @staticmethod
    # def negheaviside(x:list(T),s:float) -> np.ufunc:
    #     return cp.where(x < 0,s,0)

    @staticmethod
    def constant(x:list(T),s:float) -> np.ufunc:
        return cp.ones(len(x))
    
    @staticmethod
    def dheaviside(x:list(T),s:float) -> np.ufunc:
        return s*x*float(x==0)

    @staticmethod
    def iheaviside(x:T,s:float) -> np.ufunc:
        return x*s*float(x >= 0)
    
    @staticmethod
    #@vectorize
    def ilogit(x:float, s:float) -> cp.func:
        return 0.5*(s*cp.log(cp.cosh(x/s)) + x)
    
    @staticmethod
    #@vectorize
    def logit(x:float, s:float) -> cp.func:
        return 0.5*(1+cp.tanh(x/s))
    
    @staticmethod
    #@vectorize
    def dlogit(x:float, s:float) -> cp.func:
        return 1/(2*s*cp.cosh(x/s))
    
    @staticmethod
    #@vectorize
    def expon(x:float, s:float) -> cp.func:
        return (1 - cp.exp(-s*cp.sign(x)))
    
    @staticmethod
    #@vectorize
    def darctan(x:float,s:float) -> cp.func:
        return (s/cp.pi)*(1/(s**2 + x**2))
    
    @staticmethod
    #@vectorize
    def arctan(x:float,s:float) -> cp.func:
        return (0.5+(1/cp.pi)*cp.arctan(x/s))
    
    @staticmethod
    #@vectorize
    def iarctan(x:float,s:float) -> cp.func:
        return (x*Basis.arctan(x,s)-0.5*(s*cp.log(s**2 + x**2))/cp.pi)
    
    @staticmethod
    #@vectorize
    def sigmoid(x:float,s:float) -> cp.ufunc:
        return 1.0/(1.0+cp.exp(-s*x))
        
    @staticmethod
    #@vectorize
    def isigmoid(x:float,s:float) -> cp.ufunc:
        return cp.log(1.0+cp.exp(s*x))/s
    
    @staticmethod
    #@vectorize
    def norm(x:float,s:float) -> cp.ufunc:
        k = s*np.sqrt(2*cp.pi)
        return cp.exp(-0.5*(x/s)**2)/k

    @staticmethod
    def sinc(x:float,s:float) -> cp.ufunc:
        return cp.sinc(x*s)

    @staticmethod
    #@vectorize
    def dsigmoid(x:float,s:float) -> cp.ufunc:
        return (s*cp.exp(-s*x))/(Basis.sigmoid(x,s))**2

    def _default_base(self,x:list(T),s:float) -> cp.ufunc:
        return Basis.heaviside(x,s)
        
    def __init__(self,bfunc=None, param:float=1,lbound:float= -np.Inf,ubound:float = np.Inf) -> None:
        self.lbound = lbound
        self.ubound = ubound
        self.param = param
        
        if bfunc is None:
            self._base = Basis.heaviside
        else:
            self._base = bfunc
            
    def base(self) -> cp.ufunc:
        return self._base
    
    def integrand(self,x:T) -> cp.ufunc:
        return self._ibase(x,self.param)