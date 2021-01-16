from __future__ import annotations
import numpy as np

from typing import Union
import hotstepper.fastbase as fb

class Basis():
    T = Union[int,float]
    
    __slots__ = ('lbound','ubound','_base','_ibase','_dbase','param')

    @staticmethod
    def heaviside(x:list(T),s:float) -> np.ufunc:
        return np.where(x >= 0,s,0)

    @staticmethod
    def constant(x:list(T),s:float) -> np.ufunc:
        return np.ones(len(x))
    
    @staticmethod
    #@vectorize
    def logit(x:float, s:float) -> np.func:
        return 0.5*(1+np.tanh(x/s))
    
    @staticmethod
    #@vectorize
    def expon(x:float, s:float) -> np.func:
        return (1 - np.exp(-s*np.sign(x)))
    
    @staticmethod
    #@vectorize
    def arctan(x:float,s:float) -> np.func:
        return (0.5+(1/np.pi)*np.arctan(x/s))
    
    @staticmethod
    #@vectorize
    def sigmoid(x:float,s:float) -> np.ufunc:
        return 1.0/(1.0+np.exp(-s*x))
    
    @staticmethod
    #@vectorize
    def norm(x:float,s:float) -> np.ufunc:
        k = s*np.sqrt(2*np.pi)
        return np.exp(-0.5*(x/s)**2)/k

    @staticmethod
    def sinc(x:float,s:float) -> np.ufunc:
        return np.sinc(x*s)

    @staticmethod
    #@vectorize
    def dsigmoid(x:float,s:float) -> np.ufunc:
        return (s*np.exp(-s*x))/(Basis.sigmoid(x,s))**2

    def _default_base(self,x:list(T),s:float) -> np.ufunc:
        return Basis.heaviside(x,s)
        
    def __init__(self,bfunc=None, param:float=1,lbound:float= -np.Inf,ubound:float = np.Inf) -> None:
        self.lbound = lbound
        self.ubound = ubound
        self.param = param
        
        if bfunc is None:
            self._base = fb.Heavisidef().base
        else:
            self._base = bfunc
            
    def base(self):
        return self._base