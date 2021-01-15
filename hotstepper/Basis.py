from __future__ import annotations
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from typing import Union


class Basis():
    T = Union[int,float]
    
    __slots__ = ('lbound','ubound','_base','_ibase','_dbase','param')
    
    @staticmethod
    def heaviside(x:list(T),s:float) -> np.ufunc:
        return xp.where(x >= 0,s,0)

    @staticmethod
    def constant(x:list(T),s:float) -> np.ufunc:
        return xp.ones(len(x))
    
    @staticmethod
    #@vectorize
    def logit(x:float, s:float) -> xp.func:
        return 0.5*(1+xp.tanh(x/s))
    
    @staticmethod
    #@vectorize
    def expon(x:float, s:float) -> xp.func:
        return (1 - xp.exp(-s*xp.sign(x)))
    
    @staticmethod
    #@vectorize
    def arctan(x:float,s:float) -> xp.func:
        return (0.5+(1/xp.pi)*xp.arctan(x/s))
    
    @staticmethod
    #@vectorize
    def sigmoid(x:float,s:float) -> xp.ufunc:
        return 1.0/(1.0+xp.exp(-s*x))
    
    @staticmethod
    #@vectorize
    def norm(x:float,s:float) -> xp.ufunc:
        k = s*np.sqrt(2*xp.pi)
        return xp.exp(-0.5*(x/s)**2)/k

    @staticmethod
    def sinc(x:float,s:float) -> xp.ufunc:
        return xp.sinc(x*s)

    @staticmethod
    #@vectorize
    def dsigmoid(x:float,s:float) -> xp.ufunc:
        return (s*xp.exp(-s*x))/(Basis.sigmoid(x,s))**2

    def _default_base(self,x:list(T),s:float) -> xp.ufunc:
        return Basis.heaviside(x,s)
        
    def __init__(self,bfunc=None, param:float=1,lbound:float= -np.Inf,ubound:float = np.Inf) -> None:
        self.lbound = lbound
        self.ubound = ubound
        self.param = param
        
        if bfunc is None:
            self._base = Basis.heaviside
        else:
            self._base = bfunc
            
    def base(self) -> xp.ufunc:
        return self._base