from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import abc
from datetime import datetime
from typing import Union


valid_input_types = (int,float,pd.Timestamp,datetime)
T = Union[valid_input_types]

class AbstractStep(metaclass=abc.ABCMeta):

    __slots__ = ('_start','_start_ts','_using_dt','_weight','_end','_basis','_base','_direction','_step_np')
    
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def plot(self,plot_range:list(Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta])=None,method:str=None,smooth_factor:Union[int,float] = 1, ts_grain:Union[int,float,pd.Timedelta] = None,ax=None,where='post',**kargs):
        pass

    def __getitem__(self,x:T) ->T:
        return self.step(x)
    
    # Property access methods, best to avoid direct access of "private" properties
    @abc.abstractmethod
    def start(self) -> T:
        pass
    
    @abc.abstractmethod
    def start_ts(self) -> T:
        pass
    
    @abc.abstractmethod
    def end(self) -> AbstractStep:
        pass
    
    @abc.abstractmethod
    def weight(self) -> T:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def link_child(self, other):
        pass

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        pass
    
    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        pass
    
    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def __irshift__(self,other:T):
        pass
    
    @abc.abstractmethod
    def __ilshift__(self,other:T):
        pass

    @abc.abstractmethod
    def __rshift__(self,other:T):
        pass
    
    @abc.abstractmethod
    def __lshift__(self,other:T):
        pass

    @abc.abstractmethod    
    def __add__(self,other:T):
        pass

    @abc.abstractmethod
    def reflect(self,reflect_point:float = 0):
        pass

    @abc.abstractmethod
    def normalise(self,norm_value:float = 1):
        pass
        
    @abc.abstractmethod
    def __pow__(self,power_val:Union[int,float]):
        pass
        
    @abc.abstractmethod
    def __mul__(self,other):
        pass
                    
    @abc.abstractmethod
    def __sub__(self,other:T):
        pass

    @abc.abstractmethod
    def integrate(self,upper:T, lower:T = 0) -> T:
        pass

    @abc.abstractmethod
    def step(self,x:T) -> T:
        pass
    
    @abc.abstractmethod
    def smooth_step(self,x:list(T),smooth_factor:Union[int,float] = 1.0,smooth_basis:Basis = None) -> list(T):
        pass