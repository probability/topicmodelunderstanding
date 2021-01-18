from __future__ import annotations
import numpy as np
import pandas as pd
import abc
from datetime import datetime
from typing import Union
import hotstepper.fastbase as fb

valid_input_types = (int,float,pd.Timestamp,datetime)
T = Union[valid_input_types]

class Bases(metaclass=abc.ABCMeta):

    @staticmethod
    def heaviside():
        return fb.Heavisidef().base
        
    @staticmethod
    def heaviside_old(x:list(T),s:float, v:float, d:float) -> np.ufunc:
        return np.where(x >= 0,s,0)

    @staticmethod
    def constant(x:list(T),s:float, v:float, d:float) -> np.ufunc:
        return np.ones(len(x))
    
    @staticmethod
    def logit_old(x:list(T),s:float, v:float, d:float) -> np.func:
        return 0.5*(1+np.tanh(x/s))

    @staticmethod
    def logit():
        return fb.Logit().base
    
    @staticmethod
    def expon(x:list(T),s:float, v:float, d:float) -> np.func:
        return (1 - np.exp(-s*np.sign(x)))
    
    @staticmethod
    def arctan(x:list(T),s:float, v:float, d:float) -> np.func:
        return (0.5+(1/np.pi)*np.arctan(x/s))
    
    @staticmethod
    def sigmoid(x:float,s:float, v:float, d:float) -> np.ufunc:
        return 1.0/(1.0+np.exp(-s*x))
    
    @staticmethod
    def norm(x:list(T),s:float, v:float, d:float) -> np.ufunc:
        k = s*np.sqrt(2*np.pi)
        return np.exp(-0.5*(x/s)**2)/k

    @staticmethod
    def sinc(x:float,s:float, v:float, d:float) -> np.ufunc:
        return np.sinc(x*s)