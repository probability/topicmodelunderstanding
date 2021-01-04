from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import pandas as pd
from sortedcontainers import SortedDict
from datetime import datetime
from typing import Optional, Union

from hotstepper.Basis import Basis
from hotstepper.AbstractStep import AbstractStep

valid_input_types = (int,float,pd.Timestamp,datetime)
T = Union[valid_input_types]
S = Union[int,float,'Optional[Step]','Optional[Steps]']
    
class Step(AbstractStep):
    
    def __init__(self, start:T=None, end:T = None, weight:T = 1, basis:Basis = Basis(), use_datetime:bool = False) -> None:
        super().__init__()
        
        self._weight = weight

        self._using_dt = use_datetime

        if start is None:
            if use_datetime:
                self._start = Step.get_epoch_start()
                self._start_ts = self._start.timestamp()
            else:
                self._start = -np.inf
                self._start_ts = self._start
        else:
            self._using_dt = Step.is_date_time(start)
            self._start = start
            
            if self._using_dt:
                self._start_ts = self._start.timestamp()
            else:
                self._start_ts = self._start


        self._end = end
        self._basis = basis
        self._base = self._basis.base()

        if end is not None:
            self._end = self.link_child(end)


    def link_child(self,other:T) -> Step:
        return Step(other,end=None,weight = -1*self._weight,basis=self._basis)
    
    def rebase(self,new_basis:Basis = Basis()) -> None:
        self._basis = new_basis
        self._base = self._basis.base()
        
    def __lt__(self, other:S) -> bool:
        if type(other) is Step:
            return self._start_ts < other._start_ts
        elif type(other) is pd.Timestamp:
            return self._start_ts < other.timestamp()
        else:
            return self._start_ts < other
    
    def __gt__(self, other:S) -> bool:
        if type(other) is Step:
            return self._start_ts > other._start_ts
        elif type(other) is pd.Timestamp:
            return self._start_ts > other.timestamp()
        else:
            return self._start_ts > other
    
    def __eq__(self, other:S) -> bool:
        if type(other) is Step:
            return self._start_ts == other.start_ts() and self._weight == other.weight()
        elif type(other) is pd.Timestamp:
            return self._start_ts == other.timestamp()
        else:
            return self._start_ts == other
    
    def __getitem__(self,x:T) -> T:
        return self.step(x)

    def __call__(self,x:T) -> T:
        return self.step(x)
    
    def step(self,x:T) -> T:

        if type(x) in Step.input_types():
            x = [x]
        elif type(x) is slice:
            x = np.arange(x.start,x.stop,x.step)
            
        if len(x) > 0 and Step.is_date_time(x[0]):
            xf = cp.asarray([t.timestamp()-self._start_ts for t in x])
        else:
            xf = cp.asarray([t-self._start_ts for t in x])
            
        result = self._weight*self._base(xf,self._basis.param)
        end_st = self._end
        
        if end_st is not None and hasattr(end_st,'step') and callable(end_st.step):
            if hasattr(cp,'asnumpy'):
                cp_res = cp.asnumpy(result)
            else:
                cp_res = result

            result = np.add(cp_res,end_st.step(x))
            
        del xf
        
        if hasattr(cp,'asnumpy'):
            return cp.asnumpy(result)
        else:
            return result

    
    def smooth_step(self,x:list(T),smooth_factor:Union[int,float] = 1.0,smooth_basis:Basis = None) -> list(T):
        if smooth_basis is None:
            smooth_basis = Basis(Basis.logit,smooth_factor*len(x))
        else:
            smooth_basis = Basis(smooth_basis,smooth_factor*len(x))
            
        self.rebase(smooth_basis)
        if self._end is not None:
            self._end.rebase(smooth_basis)
        
        smoothed = self.step(x)
        self.rebase()
        
        if self._end is not None:
            self._end.rebase()

        return smoothed

    
    def _faststep(self,x:list(T)) -> list(T):
        
        if self._basis.lbound > -np.Inf or self._basis.ubound < np.Inf:
            xr = np.where((x >= self._start_ts + self._basis.lbound) & ( x <= self._start_ts + self._basis.ubound),x,0)
        else:
            xr = x
        
        res = self._weight*self._base(xr-self._start_ts,self._basis.param)
        del xr
        return res
    
    def start_ts(self) -> T:
        return self._start_ts
    
    def start(self) -> T:
        return self._start
    
    def end(self) -> Step:
        return self._end
    
    def weight(self) -> T:
        return self._weight

    def __irshift__(self,other:T) -> Step:
        t = type(other)
        if t == Step:
            new_end = None
            if self._end is not None:
                new_end = self._end._start + other._start
            return Step(self._start + other._start,end=new_end,weight=self._weight,basis=self._basis)
        else:
            new_end = None
            if self._end is not None:
                new_end = self._end._start + other
            return Step(self._start + other,end=new_end,weight=self._weight,basis=self._basis)
    
    def __ilshift__(self,other:T) -> Step:
        t = type(other)
        if t == Step:
            new_end = None
            if self._end is not None:
                new_end = self._end._start - other._start
            return Step(self._start - other._start,end=new_end,weight=self._weight,basis=self._basis)
        else:
            new_end = None
            if self._end is not None:
                new_end = self._end._start - other
            return Step(self._start - other,end=new_end,weight=self._weight,basis=self._basis)
        
    def __rshift__(self,other:T) -> Step:
        t = type(other)
        if t == Step:
            new_end = None
            if self._end is not None:
                new_end = self._end._start + other._start
            return Step(self._start + other._start,end=new_end,weight=self._weight,basis=self._basis)
        else:
            new_end = None
            if self._end is not None:
                new_end = self._end._start + other
            return Step(self._start + other,end=new_end,weight=self._weight,basis=self._basis)
    
    def __lshift__(self,other:T) -> Step:
        t = type(other)
        
        if t == Step:
            new_end = None
            if self._end is not None:
                new_end = self._end._start - other._start
            return Step(self._start - other._start,end=new_end,weight=self._weight,basis=self._basis)
        else:
            new_end = None
            if self._end is not None:
                new_end = self._end._start - other
            return Step(self._start - other,end=new_end,weight=self._weight,basis=self._basis)
    
    def __add__(self,other:T) -> S:
        t = type(other)
        if t == Step:
            from hotstepper.Steps import Steps

            st = Steps(basis=self._basis)
            st.add([self,other])
            return st
        else:
            if self._end is None:
                st = Step(start=self._start,weight=self._weight+other)
            else:
                st = Step(start=self._start,end=self._end._start,weight=self._weight+other)
            return st

    def reflect(self,reflect_point:float = 0) -> Step:
        if self._end is None:
            return Step(start=self._start,weight=reflect_point-1*self._weight)
        else:
            return Step(start=self._start,end=self._end._start,weight=reflect_point-1*self._weight)

    def normalise(self,norm_value:float = 1) -> Step:
        if self._end is None:
            return Step(start=self._start,weight=norm_value*np.sign(self._weight))
        else:
            return Step(start=self._start,end=self._end._start,weight=norm_value*np.sign(self._weight))
        
    def __pow__(self,power_val:Union[int,float]) -> Step:
        if self._end is None:
            return Step(start=self._start,weight=self._weight**power_val)
        else:
            return Step(start=self._start,end=self._end._start,weight=self._weight**power_val)

    def __floordiv__(self,other:S) -> Step:
        # result = self*other**-1
        # if result.end() is None:
        #     floor_result = Step(result.start(),np.floor(result.weight()))
        # else:
        #     floor_result = Step(result.start(),result.end().start(),np.floor(result.weight()))
        
        # return floor_result
        pass

    def __truediv__(self,other:S) -> Step:
        return self*other**-1
        
    def __mul__(self,other:S) -> Step:
        t = type(other)
        s = self
        new_weight = self._weight
        
        if t in [int,float]:
            new_weight *= other
            if self._end is None:
                return Step(start=self._start,weight=new_weight)
            else:
                return Step(start=s._start,end=self._end._start,weight=new_weight)
        
        elif t == Step:
            new_weight *= other._weight
            
            if self._end is None:
                se = np.inf
            else:
                se = self._end._start
                
            if other._end is None:
                oe = np.inf
            else:
                oe = other._end._start
                
            start = max(self._start,other._start)
            end = min(se,oe)
            
            if start < end and start != np.inf:
                if end == np.inf:
                    return Step(start=start,weight=new_weight)
                else:
                    return Step(start=start,end=end,weight=new_weight)
            else:
                return Step(start=self._start,weight=0)
                    

    def __sub__(self,other:T) -> S:
        return self + other.reflect()

    def __repr__(self) -> str:
        if self._end is None:
            return ':'.join([str(self._start),str(self._weight)])
        else:
            return ':'.join([str(self._start),str(self._end),str(self._weight)])
    
    def integrate(self,upper:T, lower:T = 0) -> T:
        return self._weight*(self._basis.integrand(upper-self._start_ts) - self._basis.integrand(lower-self._start_ts))

    def plot(self,plot_range:list(Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta])=None,method:str=None,smooth_factor:Union[int,float] = 1, ts_grain:Union[int,float,pd.Timedelta] = None,ax=None,where='post',**kargs):
        if ax is None:
            _, ax = plt.subplots()

        color = kargs.pop('color',None)
        
        if color is None:
            color=Step.get_default_plot_color()

        if plot_range is None:
            if self._start_ts == -np.inf:
                min_ts = 0
            else:
                min_ts = float(0.98*self._start_ts)

            if self._end is None:
                if self._start_ts == -np.inf:
                    max_ts = 1
                else:
                    max_ts = float(1.02*self._start_ts)
            else:
                max_ts = float(1.02*self._end._start_ts)
            
            if self._using_dt:
                if ts_grain==None:
                    #TODO: need to detect scale from step definitions
                    ts_grain = pd.Timedelta(minutes=10)
                    
                min_value = pd.Timestamp.fromtimestamp(min_ts)-ts_grain
                max_value = pd.Timestamp.fromtimestamp(max_ts)

                tsx = np.arange(min_value, max_value, ts_grain).astype(pd.Timestamp)
            else:
                if ts_grain==None:
                    ts_grain = 0.01
                
                min_value = min_ts-ts_grain
                max_value = max_ts

                tsx = np.arange(min_value, max_value, ts_grain)
        else:
            min_value = plot_range[0]
            max_value = plot_range[1]

            if self._using_dt:
                if len(plot_range) < 3:
                    #TODO: need to detect scale from step definitions
                    ts_grain = pd.Timedelta(minutes=10)
                else:
                    ts_grain = plot_range[2]

                tsx = tsx = np.arange(min_value, max_value, ts_grain).astype(pd.Timestamp)
            else:
                if len(plot_range) < 3:
                    #TODO: need to detect scale from step definitions
                    ts_grain = 0.01
                else:
                    ts_grain = plot_range[2]
                
                tsx = tsx = np.arange(min_value, max_value, ts_grain)


        if method == 'pretty':
            raw_steps = SortedDict()
            last_marker_index = None
            end_marker = True
            
            raw_steps[min_value] = 0

            raw_steps[self._start] = self._weight

            if self._end is not None:
                raw_steps[self._end.start()] = 0
                raw_steps[max_value] = 0
                last_marker_index = len(raw_steps) - 2
            else:
                raw_steps[max_value] = self._weight
                end_marker=False

            Step._prettyplot(raw_steps,plot_start=min_value,plot_start_value=0,ax=ax,end_index=last_marker_index,include_end=end_marker,**kargs)
        elif method == 'smooth':
            ax.step(tsx,self.smooth_step(tsx,smooth_factor), where=where,color=color, **kargs)
        elif method == 'function':
            ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)
        else:
            ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)

        return ax