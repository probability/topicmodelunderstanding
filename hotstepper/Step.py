from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import copy
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

        end_weight = None
        self._direction = 1
        self._basis = basis
        self._base = self._basis.base()
        self._using_dt = use_datetime or Step.is_date_time(start) or Step.is_date_time(end)

        if type(weight) in [list,tuple]:
            self._weight = weight[0]

            if len(weight) > 1:
                end_weight = weight[1]
        else:
            self._weight = weight

        if (start is not None) and (end is not None):
            self._start, self._start_ts = Step.get_keys(start, self._using_dt)
            self._end = self.link_child(end,end_weight)

        elif (start is not None) and (end is None):
            self._start, self._start_ts = Step.get_keys(start, self._using_dt)
            self._end = None
        elif (start is None) and (end is not None):
            self._direction = -1
            self._start, self._start_ts = Step.get_keys(end, self._using_dt)
            if end_weight is not None:
                self._end = self.link_child(end,end_weight)
            else:
                self._end = None
        else:
            self._basis = Basis(Basis.constant)
            self._base = self._basis.base()
            self._start, self._start_ts = Step.get_keys(start, self._using_dt,is_inf=True)
            self._end = None


    def link_child(self,other:T,weight:T = None) -> Step:
        if weight is None:
            return Step(start=other,end=None,weight = -1*self._weight)
        else:
            return Step(start=other,end=None,weight=weight)
    
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
            return self._start_ts == other._start_ts and self._weight == other._weight and self._direction == other._direction and self._end == other._end
        else:
            raise TypeError('Can only directly compare step with step object.')
        # elif type(other) is pd.Timestamp:
        #     return self._start_ts == other.timestamp()
        # else:
        #     return self._start_ts == other
    
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
            xf = cp.asarray([(t.timestamp()-self._start_ts)*self._direction for t in x])
        else:
            xf = cp.asarray([(t-self._start_ts)*self._direction for t in x])
            
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
        
        res = self._weight*self._base((xr-self._start_ts)*self._direction,self._basis.param)
        del xr
        return res
    

    def direction(self) -> T:
        return self._direction

    def start_ts(self) -> T:
        return self._start_ts
    
    def start(self) -> T:
        return self._start
    
    def end(self) -> Step:
        return self._end
    
    def weight(self) -> T:
        return self._direction*self._weight

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

            st = Steps(use_datetime=self._using_dt)
            st.add([self,other])
            return st
        else:
            if self._end is None:
                st = Step(start=self._start,weight=self._weight+other)
            else:
                st = Step(start=self._start,end=self._end._start,weight=self._weight+other)
            return st

    def reflect(self,axis:int = 0, reflect_point:float = 0) -> Step:
        reflected_step = copy.deepcopy(self)

        if axis == 0:
            Step._modify_step(reflected_step,'_weight',reflect_point-1*reflected_step._weight)

            if reflected_step._end is not None:
                Step._modify_step(reflected_step._end,'_weight',reflect_point-1*reflected_step._end._weight)
        else:
            #If the step has an end, axis=1 reflect is the same as axis=0
            if reflected_step._end is not None:
                Step._modify_step(reflected_step,'_weight',reflect_point-1*reflected_step._weight)
                Step._modify_step(reflected_step._end,'_weight',reflect_point-1*reflected_step._end._weight)
            else:
                Step._modify_step(reflected_step,'_direction',-1*reflected_step._direction)

        return reflected_step

    def normalise(self,norm_value:float = 1) -> Step:
        normed_step = copy.deepcopy(self)

        Step._modify_step(normed_step,'_weight',norm_value*np.sign(normed_step._weight))

        if normed_step._end is not None:
            Step._modify_step(normed_step._end,'_weight',norm_value*np.sign(normed_step._end._weight))

        return normed_step
        
    def __pow__(self,power_val:Union[int,float]) -> Step:
        pow_step = copy.deepcopy(self)

        Step._modify_step(pow_step,'_weight',pow_step._weight**power_val)

        if pow_step._end is not None:
            Step._modify_step(pow_step._end,'_weight',-1*np.sign(pow_step.weight())*np.abs(pow_step._end.weight()**power_val))

        return pow_step

    def __floordiv__(self,other:S) -> Step:
        """
        For now this is the same as true div, both only use the common overlap, so if demoninator step is zero in a non-zero region of the numerator, this is non-overlap and returns zero.
        """
        return self*other**-1

    def copy(self) -> Step:
        return copy.deepcopy(self)

    def __truediv__(self,other:S) -> Step:
        """
        A common overlap division, so if demoninator step is zero in a non-zero region of the numerator, this is non-overlap and returns zero.
        """

        return self*other**-1

    def __mul__(self,other:S) -> Step:
        t = type(other)
        s = self
        new_weight = self._weight

        if t in [int,float]:
            new_weight *= other

            new_step = copy.deepcopy(self)

            Step._modify_step(new_step,'_weight',new_weight)

            if new_step._end is not None:
                new_weight = new_step._end._weight*other
                Step._modify_step(new_step._end,'_weight',new_weight)

            return new_step

        elif t == Step:
            new_weight *= other._weight

            if self._end is None:
                se = np.inf
            else:
                se = self._end._start_ts

            if other._end is None:
                oe = np.inf
            else:
                oe = other._end._start_ts

            start = max(self._start_ts,other._start_ts)
            end = min(se,oe)

            if start < end and start != np.inf:
                if self._using_dt:
                    nstart = datetime.fromtimestamp(start)
                    if end != np.inf:
                        nend = datetime.fromtimestamp(end)
                    else:
                        nend = None
                else:
                    nstart = start
                    nend = end

                if end == np.inf:
                    if self._direction == -1 or other.direction() == -1:
                        new_s = Step(start=nstart,weight=new_weight,use_datetime=self._using_dt)
                        Step._modify_step(new_s,'_direction',-1)
                        return new_s
                    else:
                        return Step(start=nstart,weight=new_weight,use_datetime=self._using_dt)
                else:
                    return Step(start=nstart,end=nend,weight=new_weight,use_datetime=self._using_dt)
            else:
                return Step(start=self._start,weight=0,use_datetime=self._using_dt)


    def __sub__(self,other:T) -> S:
        return self + other.reflect()

    def __repr__(self) -> str:
        if self._end is None:
            return ':'.join([str(self._start),str(self._weight)])
        else:
            return ':'.join([str(self._start),str(self._weight),str(self._end)])
    
    def integrate(self,upper:T, lower:T = 0) -> T:
        return self._weight*(self._basis.integrand(upper-self._start_ts) - self._basis.integrand(lower-self._start_ts))

    def plot(self,plot_range:list(Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta])=None,method:str=None,smooth_factor:Union[int,float] = 1, ts_grain:Union[int,float,pd.Timedelta] = None,ax=None,where='post',**kargs):
        if ax is None:
            _, ax = plt.subplots()

        color = kargs.pop('color',None)

        if color is None:
            color=Step.get_default_plot_color()

        if self._end is None:
            max_ts = float(1.02*self._start_ts)
        else:
            max_ts = float(1.02*self._end.start_ts())

        if self._start == Step.get_epoch_start():
            min_ts = float(0.96*max_ts)
        else:
            min_ts = float(0.98*self._start_ts)

        if self._using_dt:
            if ts_grain==None:
                ts_grain = pd.Timedelta(minutes=10)
                
            min_value = datetime.fromtimestamp(min_ts)-ts_grain
            max_value = datetime.fromtimestamp(max_ts)

            tsx = np.arange(min_value, max_value, ts_grain).astype(pd.Timestamp)
        else:
            if ts_grain==None:
                ts_grain = 0.01
            
            min_value = min_ts-ts_grain
            max_value = max_ts

            tsx = np.arange(min_value, max_value, ts_grain)       
        # if color is None:
        #     color=Step.get_default_plot_color()

        # if plot_range is None:
        #     if self._start_ts == -np.inf:
        #         min_ts = 0
        #     else:
        #         min_ts = float(0.98*self._start_ts)

        #     if self._end is None:
        #         if self._start_ts == -np.inf:
        #             max_ts = 1
        #         else:
        #             max_ts = float(1.02*self._start_ts)
        #     else:
        #         max_ts = float(1.02*self._end._start_ts)
            
        #     if self._using_dt:
        #         if ts_grain==None:
        #             #TODO: need to detect scale from step definitions
        #             ts_grain = pd.Timedelta(minutes=10)
                
        #           min_value = pd.Timestamp.fromtimestamp(min_ts)-ts_grain
        #             max_value = pd.Timestamp.fromtimestamp(max_ts)

        #         tsx = np.arange(min_value, max_value, ts_grain).astype(pd.Timestamp)
        #     else:
        #         if ts_grain==None:
        #             ts_grain = 0.01
                
        #         min_value = min_ts-ts_grain
        #         max_value = max_ts

        #         tsx = np.arange(min_value, max_value, ts_grain)
        # else:
        #     min_value = plot_range[0]
        #     max_value = plot_range[1]

        #     if self._using_dt:
        #         if len(plot_range) < 3:
        #             #TODO: need to detect scale from step definitions
        #             ts_grain = pd.Timedelta(minutes=10)
        #         else:
        #             ts_grain = plot_range[2]

        #         tsx = tsx = np.arange(min_value, max_value, ts_grain).astype(pd.Timestamp)
        #     else:
        #         if len(plot_range) < 3:
        #             #TODO: need to detect scale from step definitions
        #             ts_grain = 0.01
        #         else:
        #             ts_grain = plot_range[2]
                
        #         tsx = tsx = np.arange(min_value, max_value, ts_grain)


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