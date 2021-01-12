from __future__ import annotations
from os import stat
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
try:
    import cupy as cp
except ImportError:
    import numpy as cp

import pandas as pd
import abc
from sortedcontainers import SortedDict
from datetime import datetime
from typing import Optional, Union
from hotstepper.Basis import Basis


valid_input_types = (int,float,pd.Timestamp,datetime)
T = Union[valid_input_types]

class AbstractStep(metaclass=abc.ABCMeta):
    
    @staticmethod
    def input_types():
        return valid_input_types

    @staticmethod
    def get_default_plot_color():
        return '#9c00ff'

    @staticmethod
    def get_epoch_start(use_datetime:bool = True):
        if use_datetime:
            return pd.Timestamp.min
        else:
            return -np.inf

    @staticmethod
    def get_epoch_end(use_datetime:bool = True):
        if use_datetime:
            return pd.Timestamp.max
        else:
            return np.inf

    @staticmethod
    def get_value(val, is_dt = False):
        if is_dt:
            return val.timestamp()
        else:
            return val

    @staticmethod
    def _modify_step(obj, attr:str,new_value, change_end:bool = False):
        if obj is not None and hasattr(obj,attr):
            setattr(obj,attr,new_value)

        if change_end and obj._end is not None and hasattr(obj._end,attr):
            setattr(obj,attr,new_value)

    @staticmethod
    def get_keys(val, is_dt = False, is_inf = False):

        if is_inf or val is None:
            val = AbstractStep.get_epoch_start(is_dt)

        return val, AbstractStep.get_value(val,is_dt)

    @staticmethod
    def simple_plot(xdata,ydata,cdata=None, ax=None,**kargs):
        if ax is None:
            _, ax = plt.subplots()

        dfplot = pd.DataFrame()
        dfplot['x'] = xdata
        dfplot['y'] = ydata
        
        if cdata is None:
            dfplot.plot(x='x',y='y', ax=ax, **kargs)
        else:
            dfplot['c'] = cdata
            dfplot.plot(x='x',y='y', c='c', ax=ax, **kargs)

        return ax

    @staticmethod
    def _prettyplot(step_dict:SortedDict,plot_start=0,plot_start_value=0,ax=None,start_index=1,end_index=None,include_end=True,**kargs) -> Axes:

        step0_k = plot_start
        step0_v = plot_start_value

        color = kargs.pop('color',None)
        if color is None:
            color=AbstractStep.get_default_plot_color()

        if end_index is None:
            end_index = len(step_dict)-1

        if start_index == 0:
            start_index = 1
        
        for i, (k,v) in enumerate(step_dict.items()):
            ax.hlines(y = step0_v, xmin = step0_k, xmax = k,color=color,**kargs)
            ax.vlines(x = k, ymin = step0_v, ymax = v,linestyles=':',color=color,**kargs)

            if i > start_index - 1 and i < end_index:
                if i == start_index:
                    ax.plot(k,v,marker='o',fillstyle='none',color=color,**kargs)
                else:
                    ax.plot(k,step0_v,marker='o',fillstyle='none',color=color,**kargs)
                    ax.plot(k,v,marker='o',fillstyle='none',color=color,**kargs)
            elif i == end_index and include_end:
                ax.plot(k,step0_v,marker='o',fillstyle='none',color=color,**kargs)

            step0_k = k
            step0_v = v

    @staticmethod
    def is_date_time(value:T) -> bool:
        return hasattr(value,'timestamp') and callable(value.timestamp)
    
    __slots__ = ('_start','_start_ts','_using_dt','_weight','_end','_basis','_base','_direction')
    
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

    def rebase(self,new_basis:Basis = Basis()) -> None:
        self._basis = new_basis
        self._base = self._basis.base()

    def _faststep(self,x:list(T)) -> list(T):
        
        if self._basis.lbound > -np.Inf or self._basis.ubound < np.Inf:
            xr = np.where((x >= self._start_ts + self._basis.lbound) & ( x <= self._start_ts + self._basis.ubound),x,0)
        else:
            xr = x
        
        res = self._weight*self._base(xr-self._start_ts,self._basis.param)
        del xr

        return res

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