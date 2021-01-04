from __future__ import annotations
from ast import dump

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import copy
from scipy import stats
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
from collections import defaultdict
import operator
from numpy.core.records import array
import pandas as pd
from sortedcontainers import SortedDict,SortedSet
from typing import DefaultDict, Optional, OrderedDict, Union
from datetime import datetime, timedelta

from docs.documentor import add_doc, append_doc

from hotstepper.Basis import Basis
from hotstepper.AbstractStep import AbstractStep
from hotstepper.Step import Step


V = Union[Step,'Optional[Steps]',int,float,np.float64, np.int32]

valid_input_types = (int,float,np.float64, np.int32,pd.Timestamp,datetime)
T = Union[valid_input_types]

class Analysis():
    
    def __init__(self):
        pass
    
    @staticmethod
    def pacf(st:Steps, maxlags:int = None) -> T:
        
        lags = []
        data = st.step_values()

        if (maxlags is None) or (maxlags >= len(data)):
            maxlags = len(data) - 1 
            
        pacf = np.empty(maxlags + 1)
        pacf[0] = 1.0
        
        lags = list(range(0, maxlags + 1))

        xlags, x0 = lagmat(data, maxlags, original="sep")
        xlags = add_constant(xlags)

        for lag in range(1, maxlags + 1):
            params = np.linalg.lstsq(xlags[lag:, : lag + 1], x0[lag:], rcond=None)[0]
            pacf[lag] = params[-1]
            
        return lags,pacf

    @staticmethod
    def span_and_weights(st:Steps) -> list(T):
        step_ts = np.array([s.start_ts() for s in st.steps()])
        span = np.max(step_ts) - np.min(step_ts)
        span_deltas = np.diff(step_ts)
        weights = np.divide(span_deltas,span)

        return np.min(step_ts),np.max(step_ts),span, weights

    @staticmethod
    def mean_integrate(st:Steps) -> list(T):
        steps_raw = st.steps_values()
        _,_,span, weight = Analysis.span_and_weights(st)
        mean = np.dot(steps_raw[0:-1],weight)
        var = np.dot(np.power(steps_raw[0:-1],2),weight) - mean**2
        area = span*mean

        if st.using_datetime():
            return mean,area/3600,var
        else:
            return mean,area,var

    @staticmethod
    def mean(st:Steps) -> T:
        m,a,v = Analysis.mean_integrate(st)
        return m

    @staticmethod
    def var(st:Steps) -> T:
        m,a,v = Analysis.mean_integrate(st)
        return v

    @staticmethod
    def std(st:Steps) -> T:
        m,a,v = Analysis.mean_integrate(st)
        return np.sqrt(v)
    
    @staticmethod
    def integrate(st:Steps) -> T:
        m,a,v = Analysis.mean_integrate(st)
        return a
    
    @staticmethod
    def percentile(st:Steps, percent) -> T:
        steps_raw = st.steps_values()
        return np.percentile(steps_raw,percent)

    @staticmethod
    def min(st:Steps) -> T:
        return np.min(st.steps_values())

    @staticmethod
    def max(st:Steps) -> T:
        return np.max(st.steps_values())

    @staticmethod
    def mode(st:Steps, policy='omit') -> T:
        m,c = stats.mode(st.steps_values(),nan_policy=policy)
        return m[0]

    @staticmethod
    def covariance(st:Steps,other:Steps) -> T:
        return Analysis.mean(st*other) - Analysis.mean(st)*Analysis.mean(other)
 
    @staticmethod
    def correlation(st:Steps,other:Steps) -> T:
        return Analysis.covariance(st,other)/(Analysis.std(st)*Analysis.std(other))


class Steps(AbstractStep):
    """
    The main class representing a complex step function made of individual step objects. The Steps object can be treated as a 
    mathemtical function in addition to a Python object.

    Parameters
    ----------
    use_datetime : bool ***default = False***

    Set this value to indicate that all independant variable values, the step locations are date time format, else they will be assumed to be either
    integer or float values.

    basis: Basis ***default = Basis()***
    The is the basis function that will be used for all steps associated with thie step function. The default basis -> Basis() is the Heaviside function
    $
        \theta(t) = \left\{
                \begin{array}{ll}
                    0 & \quad t < 0 \\
                    1 & \quad t \geq 0
                \end{array}
            \right.
        $
        where $t \in \mathbb{R}$

    Returns
    -------
    An instance of the Steps object with no data attached, the same as a function $f(x) = 0, \\forall x$.
    """

    
    def __init__(self,use_datetime=False,basis:Basis = Basis()) -> None:
        super().__init__()
        self._steps = np.array([],dtype=Step)
        self._basis = basis
        self._using_dt = use_datetime
        self._base = basis.base()
        self._cumsum = np.array([])
        self._cummulative = SortedDict()

        

    @staticmethod
    def aggregate(stepss: list(Optional[Steps]), aggfunc:cp.ufunc, sample_points:list(T)=None) -> Steps:
        """
        Return weights for an Np-point central derivative.
        Assumes equally-spaced function points.
        If weights are in the vector w, then
        derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)
        Parameters
        ----------
        Np : int
            Number of points for the central derivative.
        ndiv : int, optional
            Number of divisions. Default is 1.
        Returns
        -------
        w : ndarray
            Weights for an Np-point central derivative. Its size is `Np`.
        Notes
        -----
        Can be inaccurate for a large number of points.

        See Also
        --------
        Steps.

        Examples
        --------
        We can calculate a derivative value of a function.
        >>> from scipy.misc import central_diff_weights
        >>> def f(x):
        ...     return 2 * x**2 + 3
        >>> x = 3.0 # derivative point
        >>> h = 0.1 # differential step
        >>> Np = 3 # point number for central derivative
        >>> weights = central_diff_weights(Np) # weights for first derivative
        >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
        >>> sum(w * v for (w, v) in zip(weights, vals))/h
        11.79999999999998
        This value is close to the analytical solution:
        f'(x) = 4x, so f'(3) = 12
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Finite_difference
        """
        
        step_stack = np.array([])

        using_datetime = stepss[0].using_datetime()

        if sample_points is None:
            sample_points = SortedSet()
            for sts in stepss:
                sample_points.update(sts.step_keys())

        step_stack_agg = aggfunc(np.vstack((sts.step(sample_points) for sts in stepss)),axis=0)

        step0 = 0
        for i, p in enumerate(sample_points):
            if i == 0:
                step_stack = np.append(step_stack,Step(start=p,weight=step_stack_agg[i]))
                step0 = step_stack_agg[i]
            else:
                step_stack = np.append(step_stack,Step(start=p,weight=step_stack_agg[i]-step0))
                step0 = step_stack_agg[i]

        new_steps = Steps(using_datetime)

        return new_steps.add(step_stack)

    @staticmethod
    def _fill_missing(dt, fill) -> pd.Timestamp:
        if pd.isnull(dt):
            return fill
        else:
            return dt

    @staticmethod
    def read_dict(data:Union[dict,SortedDict,OrderedDict,DefaultDict],use_datetime:bool = False, convert_delta:bool = False) -> Steps:
        """
        Read a dictionary with values that represent either the cummulative value of the data steps or the direct step
        values seperately, indexed by the dictionary key values.

        Parameters
        ==============
        data : dict, SortedDict, OrderedDict, DefaultDict. 
        
        A dictionary representing the data to convert to steps.

        convert_delta : bool (default = False). 
        
        Assume values are individual step weights (default), or convert values by performing a delta between adjacent values.

        use_datetime : bool (default = False). 
        
        Assume start and end fields are of datetime format.

        convert_delta : bool, (default = False). Assume values are individual step weights (default), or convert values
        by performing a delta between adjacent values.

        Returns
        ============
        Steps : A new steps object representing the input data.

        Returns
        ============
        Steps : A new steps object representing the input data.
        
        """

        if type(data) in [dict,SortedDict,OrderedDict,DefaultDict] :
            if Steps.is_date_time(data.keys()[0]) or use_datetime:
                st = Steps(True)
            elif type(data.keys()[0]) in valid_input_types:
                st = Steps()
            else:
                raise TypeError("start data can only be intger, float or datetime")

            if convert_delta:
                ensure_sorted = SortedDict()
                ensure_sorted.update(data)

                keys = ensure_sorted.keys()
                values = ensure_sorted.values()

                delta_values = np.diff(values)
                delta_values = np.insert(delta_values,0,values[0])

                return st.add([Step(start=k,weight=v) for k,v in zip(keys,delta_values)])
            else:
                return st.add([Step(start=k,weight=v) for k,v in data.items()])
        else:
            raise TypeError("input data must be a dictionary")
        

    @staticmethod
    def read_dataframe(data:pd.DataFrame,start:str='start',end:str=None,weight:str=None,use_datetime:bool = False, convert_delta:bool = False) -> Steps:
        """
        Read a Pandas dataframe with values that represent either the cummulative value of the data steps or the direct step
        values seperately. 

        Parameters
        ==============
        data : Pandas.DataFrame. A dataframe representing the data to convert to steps.

        start : str, (default = 'start'). The name of the column containing the start data.

        end : str, (default = None). The name of the column containing the end data.

        weight : str, (default = None). The name of the column containg the step weight data.

        use_datetime : bool, (default = False). Assume start and end fields are of datetime format.

        convert_delta : bool, (default = False). Assume values are individual step weights (default), or convert values
        by performing a delta between adjacent values.

        Returns
        ============
        Steps : A new steps object representing the input data.
        
        """

        if isinstance(data,pd.DataFrame):
            if data[start].dtypes == np.dtype('datetime64[ns]') or use_datetime:
                st = Steps(True)
                if end is not None:
                    if data[end].dtypes == np.dtype('datetime64[ns]') or use_datetime:         
                        if weight is None:
                            return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),pd.Timestamp((data[start].min()).date())),end = Steps._fill_missing(pd.Timestamp(x[end]),None)),axis=1))
                        else:
                            return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),pd.Timestamp((data[start].min()).date())),end = Steps._fill_missing(pd.Timestamp(x[end]),None),weight = x[weight]),axis=1))
                    else:
                        raise TypeError("end data must be same type as start data") 
                else:
                    if weight is None:
                        return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None)),axis=1))
                    else:
                        return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None),weight=x[weight]),axis=1))
            else:# data[start].dtypes in valid_input_types:
                st = Steps(False)
                if end is not None:
                    if data[end].dtypes in valid_input_types:            
                        if weight is None:
                            return st.add(data.apply(lambda x: Step(start = x[start],end = x[end]),axis=1))
                        else:
                            return st.add(data.apply(lambda x: Step(start = x[start],end = x[end],weight = x[weight]),axis=1))
                    else:
                        raise TypeError("end data must be same type as start data") 
                else:
                    if weight is None:
                        return st.add(data.apply(lambda x: Step(start = x[start]),axis=1))
                    else:
                        return st.add(data.apply(lambda x: Step(start = x[start],weight=x[weight]),axis=1))
            #else:
            #    raise TypeError("start data can only be intger, float or datetime")    
        else:
            raise TypeError("input data must be a Dataframe")

    @staticmethod
    def read_array(start:Union[array,np.ndarray,pd.core.series.Series],end:Union[array,np.ndarray,pd.core.series.Series]=None,weight:Union[array,np.ndarray,pd.core.series.Series]=None,use_datetime:bool = False, convert_delta:bool = False) -> Steps:
        """
        Read arrays of values for start, end and weight values that represent either the cummulative value of the data steps or the direct step
        values seperately, indexed by the start and possibly end arrays.

        Parameters
        ==============
        start : array, numpy.array. An array of step start location values.

        end : array, numpy.array, (default = None). An array of step end location values.

        weight : array, numpy.array, (default = None). An array of step weight values.

        use_datetime : bool, (default = False). Assume start and end fields are of datetime format.

        convert_delta : bool, (default = False). Assume values are individual step weights (default), or convert values
        by performing a delta between adjacent values.

        Returns
        ============
        Steps : A new steps object representing the input data.
        
        """

        if len(start) > 0: # in [array,np.ndarray,pd.core.series.Series] :
            df_data = pd.DataFrame()
            params = {}
            params['convert_delta'] = convert_delta

            ls = -1

            if start is not None:
                df_data['start'] = start
                params['start'] = 'start'
                ls = len(start)

            if end is not None:
                if (len(end)-ls) !=0:
                    raise ValueError("start, end and weight arrays must be the same length",f"start array length {ls}, end array length {len(end)}")

                df_data['end'] = end
                params['end'] = 'end'

            if weight is not None:
                if (len(weight)-ls) !=0:
                    raise ValueError("start, end and weight arrays must be the same length",f"start array length {ls}, weight array length {len(weight)}")

                if convert_delta:
                    w = np.diff(weight)
                    w = np.insert(w,0,weight[0])

                    df_data['weight'] = w
                else:
                    df_data['weight'] = weight
                
                params['weight'] = 'weight'


            if Steps.is_date_time(start[0]) or use_datetime:
                params['use_datetime'] = True
                df_data.start = df_data.start.apply(pd.Timestamp)
                
                if end is not None:
                    df_data.end = df_data.end.apply(pd.Timestamp)

            #elif type(start[0]) in valid_input_types:
            #    params['use_datetime'] = False
            else:
                params['use_datetime'] = False
                #print(type(start[0]))
                #raise TypeError("start and end data can only be integer, float or datetime")

            return Steps.read_dataframe(df_data,**params)
        else:
            raise TypeError("input data must be an array")

                
    def clear(self) -> None:
        """
        Clear all the step data defined within the steps object, the same as defining a new Steps object with no data.
        """

        self._steps = np.array([],dtype=Step)
        self._cumsum = np.array([])
        self._cummulative = SortedDict()
        
    def add(self,steps:list(Step)) -> Steps:
        """
        Add an array of individual step objects to this collection of steps.

        Parameters
        ==============
        steps : Array[Steps]. Array of step objects to be added.

        Returns
        ============
        Steps : A new steps object consisting of this object and the additional steps.

        """
        
        end_steps = [s.end() for s in steps if s.end() is not None]

        self._steps = np.append(self._steps,copy.deepcopy(steps))
        self._steps = np.append(self._steps,copy.deepcopy(end_steps))

        #self._steps_ends = np.append(self._steps_ends,copy.deepcopy(end_steps))

        self._steps = np.sort(self._steps)
        self._cumsum = np.cumsum([s.weight() for s in self._steps])
        self._cummulative = self.to_dict()
        return self

    def __add__(self,other:V) -> Steps:
        """
        The '+' operation to add steps objects together like they are numbers.

        Parameters
        ==============
        other : int, float, step, steps. The thing to add to these steps, a single step or series of steps can be combined with the steps, an single int or float can also
        be added, this will be converted to a single step and added to the steps series.

        Returns
        ============
        Steps : A new steps object consisting of this object and the additional step(s).
        
        """
        combine = self.copy()
        if type(other) == Step:
            combine.add([copy.deepcopy(other)])
                
            return combine
        
        elif type(other) == Steps:
            combine.add(copy.deepcopy(other.steps()))

            return combine
        else:
            combine.add([Step(start=None,use_datetime=self._using_dt,weight=other)])
            # if self._using_dt:
            #     combine.add([Step(Steps.get_epoch_start(),weight=other)])
            # else:
            #     combine.add([Step(-np.Inf,weight=other)])

            return combine

    def __sub__(self,other:V) -> Steps:
        combine = self.copy()
        if type(other) == Step:
            if other.end() is None:
                combine.add([Step(start=other.start(),weight=-1*other.weight())])
            else:
                combine.add([Step(start=other.start(),end=other.end().start(),weight=-1*other.weight())])
                
            return combine
        
        elif type(other) == Steps:
            other_steps = other.steps()
            sub_steps = [Step(start=s.start(),weight=-1*s.weight()) for s in other_steps]
            
            combine.add(sub_steps)
            return combine
        else:
            combine.add([Step(-np.Inf,weight=-1*other)])
            return combine
    
    def using_datetime(self) -> bool:
        return self._using_dt
    
    def copy(self) -> Steps:
        return copy.deepcopy(self)
    
    def steps_values(self) -> np.ndarray:
        return self._cumsum
        
    def steps(self) -> list(Step):
        return self._steps


    def ecdf(self):
        x = np.sort(self._cumsum)
        y = np.arange(0, len(x)) / len(x)

        return x,y

    def ecdf_plot(self,ax=None,**kargs):
        x,y = self.ecdf()

        return Steps.simple_plot(x,y,ax=ax,legend=False,**kargs)

    def ecdf_step(self):
        x,y = self.ecdf()

        return Steps.read_array(start = x, weight = y, convert_delta=True)

    def histogram(self, bins=None,axis=None,ts_grain = None):
        
        """
        Calculates a histogram for the corresponding step function values
       
        Parameters
        ----------
        bins : array-like of int or float, optional
            defines the bin edges for the histogram (remember it is the step-function range that is being binned).
            If not specified the bin_edges will be assumed to be the integers which cover the step function range
        closed: {'left', 'right'}, default 'left'
            determines whether the bins, which are half-open intervals, are left-closed , or right-closed
              
        Returns
        -------
        :class:`pandas.Series`
            A Series, with a :class:`pandas.IntervalIndex`, representing the values of the histogram
            
        See Also
        --------
        Steps.ecdf
        """

        interval = 0

        if axis is None or axis == 0:
            data = self._cummulative.values()
        else:
            data = self._cummulative.keys()

        if bins is None:
            if axis is not None and axis > 0 and self._using_dt:
                interval = pd.Timedelta(minutes=1)
            else:
                interval = 1
        else:
            if axis is not None and axis > 0 and self._using_dt:
                pass
            else:
                interval = (np.max(data) - np.min(data))/bins

        l = len(data)

        if axis is not None and axis > 0 and self._using_dt:
            rang = np.arange(np.min(data),np.max(data)+interval,interval).astype(pd.Timedelta)
        else:
            rang = np.arange(np.min(data),np.max(data)+interval,interval)
        histo = {}

        for i in rang:
            histo[i] = sum(np.where((data >= i) & (data < i+interval),1,0))/l

        return list(histo.keys()), list(histo.values())


    def histogram_plot(self, bins=None,axis=None,ts_grain = None,ax=None,**kargs):

        x,y = self.histogram(bins,axis,ts_grain)

        return Steps.simple_plot(x,y,ax=ax,legend=False,**kargs)


    def histogram_step(self, bins=None,axis=None,ts_grain = None):
        
        """
        Calculates a histogram for the corresponding step function values
       
        Parameters
        ----------
        lower : int, float or pandas.Timestamp, optional
            lower bound of the step-function domain on which to perform the calculation
        upper : int, float or pandas.Timestamp, optional
            upper bound of the step-function domain to perform the calculation
        bins : array-like of int or float, optional
            defines the bin edges for the histogram (remember it is the step-function range that is being binned).
            If not specified the bin_edges will be assumed to be the integers which cover the step function range
        closed: {'left', 'right'}, default 'left'
            determines whether the bins, which are half-open intervals, are left-closed , or right-closed
              
        Returns
        -------
        :class:`pandas.Series`
            A Series, with a :class:`pandas.IntervalIndex`, representing the values of the histogram
            
        See Also
        --------
        Steps.ecdf
        """

        x,y = self.histogram(bins,axis,ts_grain)

        return Steps.read_array(start=x, weight=y,convert_delta=True)


    def rebase(self,new_basis:Basis = Basis(),change_steps=False) -> None:
        if change_steps:
            self._basis = new_basis
            self._base = new_basis.base()
        else:
            for s in self._steps:
                if s._base is not Basis.constant:
                    s.rebase(new_basis)

    def clip(self,lbound:T=None,ubound:T=None) -> Steps:
        
        if self._using_dt:
            delta = pd.Timedelta(1,unit='ns')
        else:
            delta = 0.000000001

        data = self.to_dict(False)

        if lbound is None and ubound is None:
            return self
        elif lbound is None:
            new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k <= ubound)])
            
            clip_end = (self.step(ubound-delta))[0]
            new_steps = np.append(new_steps,Step(start=ubound,weight=-1*clip_end))

        elif ubound is None:
            new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k >= lbound)])

            clip_start = (self.step(lbound+delta))[0]
            new_steps = np.append(new_steps,Step(start=lbound,weight=clip_start))

        else:
            new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k >= lbound) and (k <= ubound)])

            clip_start = (self.step(lbound+delta))[0]
            clip_end = (self.step(ubound-delta))[0]

            new_steps = np.append(new_steps,Step(start=lbound,weight=clip_start))
            new_steps = np.append(new_steps,Step(start=ubound,weight=-1*clip_end))
        
        new_clipped = Steps(self.using_datetime())

        return new_clipped.add(new_steps)
        

    def weights(self):
        _,_,_, w = Analysis.span_and_weights(self)

        return w

    def span(self):
        smin,smax,s, _ = Analysis.span_and_weights(self)

        return smin,smax,s

    def reduce(self, full_reduce:bool=False) -> None:      

        data = self.to_dict(False)
            
        new_steps = [Step(start=k,weight=v) for k,v in data.items() if v != 0]
        
        self.clear()
        self.add(new_steps)


    def step_values(self) -> list(T):
        return self._cumsum
    
    def step_keys(self) -> list(T):
        return list(self._cummulative.keys())
    
    
    def to_dict(self,use_cummulative:bool = True, only_ends:bool = False) -> SortedDict:
        data:defaultdict = defaultdict(lambda:0)
        
        if use_cummulative:
            for s, cs in zip(self._steps,self._cumsum):
                data[s.start()] = cs
        else:
            for s in self._steps:
                if only_ends and s.end() is not None:
                    data[s.end()] += s.weight()                    
                else:
                    data[s.start()] += s.weight()

        return SortedDict(data)
    
    def to_dataframe(self,mode:str = 'aggregate') -> pd.DataFrame:

        if mode in ['aggregate','cummulative']:
            data = self.to_dict(mode == 'cummulative')
            return pd.DataFrame.from_dict({'start': list(data.keys()), 'value': list(data.values())})
        elif mode == 'full':
            data:array = []
            ends:array = []

            for s in self._steps:
                if s.end() is not None:
                    ends.append(s.end())
                    data.append({'start': s.start(),'end':s.end().start(),'value':s.weight()})                  
                else:
                    if s not in ends:
                        data.append({'start': s.start(),'end':None,'value':s.weight()})

            return pd.DataFrame.from_dict(data)


    def __getitem__(self,x:T) -> T:
        return self.step(x)

    def __call__(self,x:T) -> T:
        return self.step(x)
    
    def step(self, x:T) -> list(T):
        if type(x) in Step.input_types():
            x = [x]
        elif type(x) is slice:
            if Steps.is_date_time(x.start):
                if x.step is None:
                    x = np.arange(x.start,x.stop,pd.Timedelta(minutes=1)).astype(pd.Timestamp)
                else:
                    x = np.arange(x.start,x.stop,x.step).astype(pd.Timestamp)
            else:
                x = np.arange(x.start,x.stop,x.step)
            
        if len(x) > 0 and Steps.is_date_time(x[0]):
            xts = cp.array([t.timestamp() for t in x])
        else:
            xts = cp.array(x)
            
        # bottle neck is right here!
        if len(self._steps) > 0:
            stvals = cp.array([s._faststep(xts) for s in self._steps])
        else:
            if hasattr(cp,'asnumpy'):
                return cp.asnumpy(cp.zeros(len(x)))
            else:
                return cp.zeros(len(x))
            
        
        del xts
        result = cp.sum(stvals,axis=0)
        del stvals

        if hasattr(cp,'asnumpy'):
            return cp.asnumpy(result)
        else:
            return result


    def smooth_plot(self,smooth_factor:Union[int,float] = None, ts_grain:Union[int,float,pd.Timedelta] = None,ax=None,where='post',**kargs):
        return self.plot(method='smooth',smooth_factor=smooth_factor, ts_grain=ts_grain,ax=ax,where=where,**kargs)

    def plot(self,plot_range:list(Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta],Union[int,float,pd.Timedelta])=None,method:str=None, plot_start=None,plot_end=None,smooth_factor:Union[int,float] = None, ts_grain:Union[int,float,pd.Timedelta] = None,ax=None,where='post',**kargs):
        if ax is None:
            _, ax = plt.subplots()
        
        color = kargs.pop('color',None)
        if color is None:
            color=Steps.get_default_plot_color()

        if method == None:
            raw_steps = self.to_dict()
            
            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(seconds=1)
            else:
                ts_grain = 0.00001
                
            zero_key = (raw_steps.keys())[0] - ts_grain
            raw_steps[zero_key] = 0
            ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)

        elif method == 'pretty':
            raw_steps = self.to_dict()
            
            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(seconds=1)
            else:
                ts_grain = 0.00001

            zero_key = (raw_steps.keys())[0] - ts_grain
            raw_steps[zero_key] = 0

            Steps._prettyplot(raw_steps,plot_start=zero_key,plot_start_value=0,ax=ax,color=color,**kargs)

        elif method == 'function':
            step_ts = np.array([s.start_ts() for s in self._steps])
            max_ts = np.amax(step_ts)
            min_ts = np.amin(step_ts)
            
            if self._using_dt:
                if ts_grain==None:
                    ts_grain = pd.Timedelta(minutes=10)
                
                tsx = np.arange(pd.Timestamp.fromtimestamp(min_ts)-ts_grain, pd.Timestamp.fromtimestamp(max_ts), ts_grain).astype(pd.Timestamp)
                ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)
            else:
                if ts_grain==None:
                    ts_grain = 0.01
                
                tsx = np.arange(min_ts-ts_grain, max_ts, ts_grain)
                ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)
                
        elif method == 'smooth':
            step_ts = np.array([s.start_ts() for s in self._steps])
            max_ts = np.amax(step_ts)
            min_ts = np.amin(step_ts)

            if smooth_factor is None:
                smooth_factor = (max_ts - min_ts)/250
            
            if self._using_dt:
                if ts_grain==None:
                    ts_grain = pd.Timedelta(minutes=10)
                
                tsx = np.arange(pd.Timestamp.fromtimestamp(min_ts)-ts_grain, pd.Timestamp.fromtimestamp(max_ts), ts_grain).astype(pd.Timestamp)
            else:
                if ts_grain==None:
                    ts_grain = 0.00001
                
                tsx = np.arange(min_ts-ts_grain, max_ts, ts_grain)
            
            ax.step(tsx,self.smooth_step(tsx,smooth_factor = smooth_factor), where=where,color=color, **kargs)
        else:
            raw_steps = self.to_dict()
            
            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(seconds=1)
            else:
                ts_grain = 0.01
                
            zero_key = (raw_steps.keys())[0] - ts_grain
            raw_steps[zero_key] = 0
            ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)

        return ax
    
    def _operate_norm(self,other:V, op_func) -> Steps:
        """
        This function is used to create a normalised representation of the steps that results from applying the comparison function
        to the cummulative total of the steps.

        Parameters
        ===========
        other : int, float, Step, Steps. Any value to compare each step component against.
        op_func : Numpy Universal Function. A binary comparison function that returns a bool, e.g >,<,==.

        Returns:
        ==========
        Steps: A new steps instance with weight of 1 everywhere the filter condition was true.
        
        """

        if type(other) in [float,int]:
            new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
            new_steps = []
            
            mask = np.where(op_func(self._cumsum,other), True,False)
            
            first = True
            st = None
            all_true=True
            for i ,s in enumerate(self._steps):
                if mask[i]:
                    if first:
                        st = s
                        first=False
                        new_steps.append(Step(start=st.start(),weight=1))
                        continue
                else:
                    all_true = False
                    if not first:
                        first=True
                        new_steps.append(Step(start=s.start(),weight=-1))
                        st = None

            if all_true:
                return new_instance.add([Step(start=st.start(),end=self._steps[-1].start(),weight=1),Step(start=self._steps[-1].start(),weight=1)])
            else:
                last_step = new_steps[-1]
                new_steps.append(Step(start=self._steps[-1].start(),weight=-1*last_step.weight()))
                new_steps.append(Step(start=self._steps[-1].start(),weight=last_step.weight()))

            new_instance.add(new_steps)
            return new_instance

    def _operate_value(self,other:V, op_func) -> Steps:
        
        if type(other) in [float,int]:
            new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
            new_steps = []
            mid_steps = []
            
            # we use cumsum to ensure we have the same number of values as steps,
            # since some steps can start at the same point
            mask = np.where(op_func(self._cumsum,other), True,False)
            
            first = True
            st = None
            adj = 0
            all_true = True
            for i ,s in enumerate(self._steps):
                if mask[i]:
                    if first:
                        st = s
                        first=False
                        new_steps.append(Step(start=st.start(),weight=self._cummulative[st.start()]))
                        continue
                    elif not (st is None) and (s.start_ts() > st.start_ts()):
                        mid_steps.append(Step(start=s.start(),weight=s.weight()))
                        adj += s.weight()
                else:
                    all_true = False
                    if not first:
                        first=True
                        # the end of the big new step where the condition was met, we use adj to correct for changes
                        # since the start of the step when we want to see the steps along the way, to ensure we return
                        # to zero when condition is not met.
                        new_steps.append(Step(start=s.start(),weight=-1*(self._cummulative[st.start()] + adj)))
                        adj = 0
                        st = None
            
            if all_true:
                return self

            new_instance.add(new_steps)
            new_instance.add(mid_steps)
            return new_instance

    def reflect(self,reflect_point:float = 0) -> Steps:
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        ends = [s.end() for s in self._steps if s.end() is not None]
        
        reflected_steps = [s.reflect(reflect_point) for s in self._steps if s not in ends]
        new_instance.add(reflected_steps)
        
        return new_instance
    
    def __pow__(self,power_val:Union[int,float]) -> Steps:
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        ends = [s.end() for s in self._steps if s.end() is not None]
        
        pow_steps = [s**power_val for s in self._steps if s not in ends]
        new_instance.add(pow_steps)
        
        return new_instance

    def __iter__(self):
        self._index = 0
        return iter(self._steps)

    def __next__(self):
        if self._index < len(self._steps):
            self._index += 1
            return self._steps[self._index - 1]
        else:
            self._index = 0
            raise StopIteration

    def normalise(self) -> Steps:
        return self._operate_norm(0, operator.ne)
    
    def invert(self) -> Steps:
        return self._operate_norm(0, operator.eq)
        
    def __gt__(self,other:V) -> Steps:
        return self._operate_value(other, operator.gt)
    
    def __lt__(self,other:V) -> Steps:
        return self._operate_value(other, operator.lt)

    def __ge__(self,other:V) -> Steps:
        return self._operate_value(other, operator.ge)
    
    def __le__(self,other:V) -> Steps:
        return self._operate_value(other, operator.le)

    def __ne__(self,other:V) -> Steps:
        return self._operate_value(other, operator.ne)

    def __eq__(self,other:V) -> Steps:
        return self._operate_value(other, operator.eq)
        
    def __lshift__(self,other:V) -> Steps:
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        ends = [s.end() for s in self._steps if s.end() is not None]
        
        lshift_steps = [s<<other for s in self._steps if s not in ends]
        new_instance.add(lshift_steps)
        return new_instance
        
    def __rshift__(self,other:V) -> Steps:
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        ends = [s.end() for s in self._steps if s.end() is not None]
        
        rshift_steps = [s>>other for s in self._steps if s not in ends]
        new_instance.add(rshift_steps)
        return new_instance

    def __floordiv__(self,other:V) -> Steps:
        pass

    def rotate(self) -> Steps:
        return Steps.read_array(self.step_values(),self.step_keys(),convert_delta=True)


    def __truediv__(self,other:V) -> Steps:
        return self*other**-1

    def __mul__(self,other:V) -> Steps:
        if isinstance(other, Steps):
            new_steps = np.array([],dtype=Step)

            #Need to remove the inserted end steps as the start step parent will handle the end in the Step multiplication
            end_steps = [s.end() for s in self._steps if s.end() is not None]
            other_end_steps = [s.end() for s in other.steps() if s.end() is not None]

            for s in self._steps:
                if s not in end_steps:
                    for s_other in other.steps():
                        if s_other not in other_end_steps:
                            new_steps = np.append(new_steps,s*s_other)

            st = Steps().add(new_steps)
            st.reduce()

            return st

        else:
            new_steps = np.array([],dtype=Step)

            #Need to remove the inserted end steps as the start step parent will handle the end in the Step multiplication
            end_steps = [s.end() for s in self._steps if s.end() is not None]

            for s in self._steps:
                if s not in end_steps:
                    new_steps = np.append(new_steps,s*other)

            st = Steps().add(new_steps)
            st.reduce()

            return st

    def start_ts(self) -> T:
        pass
    
    def start(self) -> T:
        pass
    
    def end(self) -> Step:
        pass
    
    def weight(self) -> T:
        pass

    def __repr__(self) -> str:
        return ','.join([str(s) for s in self._steps])

    def link_child(self, other):
        pass
    
    def __irshift__(self,other:T):
        pass
    
    def __ilshift__(self,other:T):
        pass
    
    def smooth_step(self,x:list(T),smooth_factor:Union[int,float] = None,smooth_basis:Basis = None) -> list(T):

        step_ts = np.array([s.start_ts() for s in self._steps])
        max_ts = np.amax(step_ts)
        min_ts = np.amin(step_ts)

        if smooth_factor is None:
            smooth_factor = (max_ts - min_ts)/250

        if smooth_basis is None:
            smooth_basis = Basis(Basis.logit,smooth_factor)
        else:
            smooth_basis = Basis(smooth_basis,smooth_factor)
            
        self.rebase(smooth_basis)
        smoothed = self.step(x)
        self.rebase()

        return smoothed
                                                            
        
    def integrate(self):
        return Analysis.integrate(self)
    
    def mean(self):
        return Analysis.mean(self)
    
    def var(self):
        return Analysis.var(self)

    def min(self):
        return Analysis.min(self)

    def max(self):
        return Analysis.max(self)

    def mode(self):
        return Analysis.mode(self)
    
    def median(self):
        return Analysis.percentile(self, 50)
    
    def pacf(self, maxlags:int = None):
        l = len(self._cumsum)

        if (maxlags is None) or (maxlags >= l):
            maxlags = int(0.1*l) 

        return Analysis.pacf(self, maxlags)

    def pacf_step(self, maxlags:int = None):
        lags, pac = self.pacf(maxlags)

        return Steps.read_array(start=lags, weight=pac, convert_delta=True)

    def pacf_plot(self, maxlags:int = None,ax=None,**kargs):
        lags, pac = self.pacf(maxlags)

        return Steps.simple_plot(lags,pac,ax=ax,legend=False,**kargs)

    def percentile(self,percent):
        return Analysis.percentile(self,percent)

    def covariance(self,other:Steps) -> T:
        return Analysis.covariance(self,other)

    def correlation(self,other:Steps) -> T:
        return Analysis.correlation(self,other)