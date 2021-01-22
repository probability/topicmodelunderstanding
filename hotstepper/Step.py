from __future__ import annotations
from hotstepper.Utils import Utils
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from sortedcontainers import SortedDict
from datetime import datetime

from hotstepper.Utils import Utils
from hotstepper.Basis import Basis
from hotstepper.Bases import Bases
from hotstepper.AbstractStep import AbstractStep

valid_input_types = (int,float,pd.Timestamp,datetime)

class Step(AbstractStep):
    
    def __init__(self, start=None, end = None, weight = 1, basis = Basis(), use_datetime = False):
        super().__init__()

        end_weight = None
        self._direction = 1
        self._basis = basis
        self._base = self._basis.base()
        self._using_dt = use_datetime or Utils.is_date_time(start) or Utils.is_date_time(end)

        if type(weight) in [list,tuple]:
            self._weight = weight[0]

            if len(weight) > 1:
                end_weight = weight[1]
        else:
            self._weight = weight

        if (start is not None) and (end is not None):
            self._start, self._start_ts = Utils.get_keys(start, self._using_dt)
            self._end = self.link_child(end,end_weight)

        elif (start is not None) and (end is None):
            self._start, self._start_ts = Utils.get_keys(start, self._using_dt)
            self._end = None
        elif (start is None) and (end is not None):
            self._direction = -1
            self._start, self._start_ts = Utils.get_keys(end, self._using_dt)
            if end_weight is not None:
                self._end = self.link_child(end,end_weight)
            else:
                self._end = None
        else:
            self._basis = Basis(Bases.fconstant)
            self._base = self._basis.base()
            self._start, self._start_ts = Utils.get_keys(start, self._using_dt,is_inf=True)
            self._end = None

        self._step_np = np.array([self._start,float(self._start_ts),float(self._direction),float(self._weight)])

    def step_np(self):
        return self._step_np

    def _faststep(self,x):
        res = self._weight*self._base((x-self.start_ts())*self._direction,self._basis.param)
        return res

    def link_child(self,other,weight = None):
        return Step(start=other,end=None,weight = -1*self._weight)

    def detach_child(self):
        return self.copy(False)
    
    def rebase(self,new_basis = Basis()) -> None:
        self._basis = new_basis
        self._base = self._basis.base()
        
    def __lt__(self, other):
        if type(other) is Step:
            return self._start_ts < other._start_ts
        elif type(other) is pd.Timestamp:
            return self._start_ts < other.timestamp()
        else:
            return self._start_ts < other
    
    def __gt__(self, other):
        if type(other) is Step:
            return self._start_ts > other._start_ts
        elif type(other) is pd.Timestamp:
            return self._start_ts > other.timestamp()
        else:
            return self._start_ts > other
    
    def __eq__(self, other):

        if type(other) is Step:
            return self._start_ts == other._start_ts and self._weight == other._weight and self._direction == other._direction and self._end == other._end
        else:
            raise TypeError('Can only directly compare step with step object.')
    
    def __getitem__(self,x):
        return self.step(x)

    def __call__(self,x):
        return self.step(x)

    def step(self,x):

        if not hasattr(x,'__iter__'):
            x = np.array([x])
        elif type(x) is slice:
            x = np.arange(x.start,x.stop,x.step)

        if self._using_dt:
            x = np.asarray(list(map(Utils.get_ts, x)))

        result = self._base(x,self._start_ts,self._direction,self._weight)
        end_st = self._end
        
        if end_st is not None and hasattr(end_st,'step') and callable(end_st.step):
            result = np.add(result,end_st.step(x))

        return result


    def smooth_step(self,x,smooth_factor = None,smooth_basis = None):
        if not hasattr(x,'__iter__'):
            x = np.array([x])
        elif type(x) is slice:
            if Utils.is_date_time(x.start):
                if x.step is None:
                    x = np.arange(x.start,x.stop,pd.Timedelta(minutes=1)).astype(pd.Timestamp)
                else:
                    x = np.arange(x.start,x.stop,x.step).astype(pd.Timestamp)
            else:
                x = np.arange(x.start,x.stop,x.step)

        if smooth_factor is None:
            smooth_factor = float(len(x)*10)

        # bottle neck is right here!
        if self._using_dt:
            x = np.asarray(list(map(Utils.get_ts, x)),dtype=float)
        

        print(smooth_factor)


        result = Bases.fflogit(x,np.array([self._step_np[[1,2,3]]],dtype=float),smooth_factor)

        return result


    def direction(self):
        return self._direction

    def start_ts(self):
        return self._start_ts
    
    def start(self):
        return self._start
    
    def end(self):
        return self._end
    
    def weight(self):
        return self._direction*self._weight

    def __irshift__(self,other):
        return self.__rshift__(other)
    
    def __ilshift__(self,other):
        return self.__lshift__(other)
        
    def __rshift__(self,other):
        rshift_step = self.copy()
  
        new_end = None
        if type(other) == Step:
            if rshift_step.start() == Utils.get_epoch_end():
                new_start, new_start_ts = Utils.get_keys(rshift_step.start(), rshift_step._using_dt)
            else:
                new_start, new_start_ts = Utils.get_keys(rshift_step.start() + other, rshift_step._using_dt)

            if rshift_step._end is not None:
                new_end, new_end_ts = Utils.get_keys(rshift_step.end().start() + other, rshift_step._using_dt)
                Utils._modify_step(rshift_step._end,'_start',new_end)
                Utils._modify_step(rshift_step._end,'_start_ts',new_end_ts)

            Utils._modify_step(rshift_step,'_start',new_start)
            Utils._modify_step(rshift_step,'_start_ts',new_start_ts)
        else:
            if rshift_step.start() == Utils.get_epoch_end():
                new_start, new_start_ts = Utils.get_keys(rshift_step.start(), rshift_step._using_dt)
            else:
                new_start, new_start_ts = Utils.get_keys(rshift_step.start() + other, rshift_step._using_dt)
            
            if rshift_step._end is not None:
                new_end, new_end_ts = Utils.get_keys(rshift_step.end().start() + other, rshift_step._using_dt)
                Utils._modify_step(rshift_step._end,'_start',new_end)
                Utils._modify_step(rshift_step._end,'_start_ts',new_end_ts)

            Utils._modify_step(rshift_step,'_start',new_start)
            Utils._modify_step(rshift_step,'_start_ts',new_start_ts)

        return rshift_step
    

    def __lshift__(self,other):
        lshift_step = self.copy()
  
        new_end = None
        new_end_ts = None

        if type(other) == Step:
            if lshift_step.start() == Utils.get_epoch_start():
                new_start, new_start_ts = Utils.get_keys(lshift_step.start(), lshift_step._using_dt)
            else:
                new_start, new_start_ts = Utils.get_keys(lshift_step.start() - other, lshift_step._using_dt)
            
            if lshift_step.end() is not None:
                new_end, new_end_ts = Utils.get_keys(lshift_step.end().start() - other, lshift_step._using_dt)
                Utils._modify_step(lshift_step._end,'_start',new_end)
                Utils._modify_step(lshift_step._end,'_start_ts',new_end_ts)

            Utils._modify_step(lshift_step,'_start',new_start)
            Utils._modify_step(lshift_step,'_start_ts',new_start_ts)
        else:
            if lshift_step.start() == Utils.get_epoch_start():
                new_start, new_start_ts = Utils.get_keys(lshift_step.start(), lshift_step._using_dt)
            else:
                new_start, new_start_ts = Utils.get_keys(lshift_step.start() - other, lshift_step._using_dt)
            
            if lshift_step.end() is not None:
                new_end, new_end_ts = Utils.get_keys(lshift_step.end().start() - other, lshift_step._using_dt)
                Utils._modify_step(lshift_step._end,'_start',new_end)
                Utils._modify_step(lshift_step._end,'_start_ts',new_end_ts)

            Utils._modify_step(lshift_step,'_start',new_start)
            Utils._modify_step(lshift_step,'_start_ts',new_start_ts)

        return lshift_step

    
    def __add__(self,other):
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

    def reflect(self,axis = 0, reflect_point = 0):
        reflected_step = self.copy()

        if axis == 0:
            Utils._modify_step(reflected_step,'_weight',reflect_point-1*reflected_step._weight)

            if reflected_step._end is not None:
                Utils._modify_step(reflected_step._end,'_weight',reflect_point-1*reflected_step._end._weight)
        else:
            #If the step has an end, axis=1 reflect is the same as axis=0
            if reflected_step._end is not None:
                Utils._modify_step(reflected_step,'_weight',reflect_point-1*reflected_step._weight)
                Utils._modify_step(reflected_step._end,'_weight',reflect_point-1*reflected_step._end._weight)
            else:
                Utils._modify_step(reflected_step,'_direction',-1*reflected_step._direction)

        return reflected_step

    def normalise(self,norm_value = 1):
        normed_step = self.copy()

        Utils._modify_step(normed_step,'_weight',norm_value*np.sign(normed_step._weight))

        if normed_step._end is not None:
            Utils._modify_step(normed_step._end,'_weight',norm_value*np.sign(normed_step._end._weight))

        return normed_step
        
    def __pow__(self,power_val):
        pow_step = self.copy()

        Utils._modify_step(pow_step,'_weight',pow_step._weight**power_val)

        if pow_step._end is not None:
            Utils._modify_step(pow_step._end,'_weight',-1*np.sign(pow_step.weight())*np.abs(pow_step._end.weight()**power_val))

        return pow_step

    def __floordiv__(self,other):
        """
        For now this is the same as true div, both only use the common overlap, so if demoninator step is zero in a non-zero region of the numerator, this is non-overlap and returns zero.
        """
        return self*other**-1

    def copy(self,copy_end = True):
        new_step = None
        if self._direction == 1:
            if self._end is None or not copy_end:
                new_step = Step(start=self._start,weight=self._weight,basis=self._basis)
            else:
                new_step = Step(start=self._start, end=self._end.start(),weight=self._weight,basis=self._basis)
        else:
            new_step = Step(end=self._start,weight=self._weight,basis=self._basis)

        return new_step

    def __truediv__(self,other):
        """
        A common overlap division, so if demoninator step is zero in a non-zero region of the numerator, this is non-overlap and returns zero.
        """

        return self*other**-1

    def __mul__(self,other):
        t = type(other)
        s = self
        new_weight = self._weight

        if t in [int,float]:
            new_weight *= other

            new_step = self.copy()

            Utils._modify_step(new_step,'_weight',new_weight)

            if new_step._end is not None:
                new_weight = new_step._end._weight*other
                Utils._modify_step(new_step._end,'_weight',new_weight)

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
                    nstart = pd.Timestamp.utcfromtimestamp(start)
                    if end != np.inf:
                        nend = pd.Timestamp.utcfromtimestamp(end)
                    else:
                        nend = None
                else:
                    nstart = start
                    nend = end

                if end == np.inf:
                    if self._direction == -1 or other.direction() == -1:
                        new_s = Step(start=nstart,weight=new_weight,use_datetime=self._using_dt)
                        Utils._modify_step(new_s,'_direction',-1)
                        return new_s
                    else:
                        return Step(start=nstart,weight=new_weight,use_datetime=self._using_dt)
                else:
                    return Step(start=nstart,end=nend,weight=new_weight,use_datetime=self._using_dt)
            else:
                return Step(start=self._start,weight=0,use_datetime=self._using_dt)


    def __sub__(self,other):
        return self + other.reflect()

    def __repr__(self):
        if self._end is None:
            return ':'.join([str(self._start),str(self._weight)])
        else:
            return ':'.join([str(self._start),str(self._weight),str(self._end)])
    
    def integrate(self,upper, lower = 0):
        return self._weight*(self._basis.integrand(upper-self._start_ts) - self._basis.integrand(lower-self._start_ts))

    def smooth_plot(self,plot_range=None,method:str=None,smooth_factor = None, ts_grain = None,ax=None,where='post',**kargs):
        return self.plot(method='smooth',smooth_factor=smooth_factor, ts_grain=ts_grain,ax=ax,where=where,**kargs)

    def plot(self,plot_range=None,method:str=None,smooth_factor = None, ts_grain = None,ax=None,where='post',**kargs):
        if ax is None:
            _, ax = plt.subplots()

        color = kargs.pop('color',None)

        if color is None:
            color=Utils.get_default_plot_color()

        if self._end is None:
            max_ts = float(1.1*self._start_ts)
        else:
            max_ts = float(1.1*self._end.start_ts())

        if self._start == Utils.get_epoch_start(self._using_dt):
            min_ts = float(0.9*max_ts)
        else:
            min_ts = float(0.9*self._start_ts)

        if self._using_dt:
            if ts_grain==None:
                ts_grain = pd.Timedelta(minutes=1)
                
            min_value = pd.Timestamp.utcfromtimestamp(min_ts)
            max_value = pd.Timestamp.utcfromtimestamp(max_ts)

        else:
            if ts_grain==None:
                ts_grain = 0.01
            
            min_value = min_ts-ts_grain
            max_value = max_ts

        end_start = self._end._start if self._end is not None else None
        tsx = Utils.get_plot_range(self._start,end_start,ts_grain,use_datetime=self._using_dt)

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

            Utils._prettyplot(raw_steps,plot_start=min_value,plot_start_value=0,ax=ax,end_index=last_marker_index,include_end=end_marker,**kargs)
        elif method == 'smooth':
            ax.step(tsx,self.smooth_step(tsx,smooth_factor), where=where,color=color, **kargs)
        elif method == 'function':
            ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)
        else:
            ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)

        return ax