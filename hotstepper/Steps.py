from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
from collections import defaultdict
import operator
from itertools import groupby
from numpy.core.records import array
import pandas as pd
from sortedcontainers import SortedDict,SortedSet
from typing import DefaultDict, Optional, OrderedDict, Union
from datetime import datetime, timedelta

#from docs.documentor import add_doc, append_doc

from hotstepper.Utils import Utils
from hotstepper.Basis import Basis
from hotstepper.Bases import Bases
import hotstepper.fastbase as fb
from hotstepper.AbstractStep import AbstractStep
from hotstepper.Step import Step

valid_input_types = (int,float,np.float64, np.int32,datetime,datetime)

class Analysis():
    
    def __init__(self):
        pass
    
    @staticmethod
    def pacf(st, maxlags = None):
        
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
    def span_and_weights(st):
        if st.using_datetime():
            step_ts = np.array([k.timestamp() for k in st.step_keys()])
        else:
            step_ts = np.array([k for k in st.step_keys()])

        span = np.max(step_ts) - np.min(step_ts)
        span_deltas = np.diff(step_ts)
        weights = np.divide(span_deltas,span)

        return np.min(step_ts),np.max(step_ts),span, weights

    @staticmethod
    def mean_integrate(st):
        steps_raw = st.step_values()
        _,_,span, weight = Analysis.span_and_weights(st)
        mean = np.dot(steps_raw[0:-1],weight)
        var = np.dot(np.power(steps_raw[0:-1],2),weight) - mean**2
        area = span*mean

        if st.using_datetime():
            return mean,area/3600,var
        else:
            return mean,area,var

    @staticmethod
    def mean(st):
        m,a,v = Analysis.mean_integrate(st)
        return m

    @staticmethod
    def var(st):
        m,a,v = Analysis.mean_integrate(st)
        return v

    @staticmethod
    def std(st):
        m,a,v = Analysis.mean_integrate(st)
        return np.sqrt(v)
    
    @staticmethod
    def integrate(st):
        m,a,v = Analysis.mean_integrate(st)
        return a
    
    @staticmethod
    def percentile(st, percent):
        steps_raw = st.step_values()
        return np.percentile(steps_raw,percent)

    @staticmethod
    def min(st):
        return np.min(st.step_values())

    @staticmethod
    def max(st):
        return np.max(st.step_values())

    @staticmethod
    def mode(st, policy='omit'):
        m,c = stats.mode(st.step_values(),nan_policy=policy)
        return m[0]

    @staticmethod
    def covariance(st,other):
        return Analysis.mean(st*other) - Analysis.mean(st)*Analysis.mean(other)
 
    @staticmethod
    def correlation(st,other):
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
        \\theta(t) = \\left\{
                \\begin{array}{ll}
                    0 & \\quad t < 0 \\
                    1 & \\quad t \geq 0
                \\end{array}
            \\right.
        $
        where $t \in \mathbb{R}$

    Returns
    -------
    An instance of the Steps object with no data attached, the same as a function $f(x) = 0, \\forall x$.
    """
    
    def __init__(self,use_datetime=False,basis = Basis()):
        super().__init__()
        self._truesteps = np.array([],dtype=Step)
        self._steps = np.array([],dtype=Step)
        self._step_np = np.empty([1,4])
        self._basis = basis
        self._using_dt = use_datetime
        self._base = basis.base()
        self._cummulative = SortedDict()


    @staticmethod
    def aggregate(stepss, aggfunc, sample_points=None):
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
    def _fill_missing(dt, fill):
        if pd.isna(dt):
            return fill
        else:
            return dt

    @staticmethod
    def read_dict(data,use_datetime = False, convert_delta = True):
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
            if Steps.is_date_time(list(data.keys())[0]) or use_datetime:
                st = Steps(True)
            elif type(list(data.keys())[0]) in valid_input_types:
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
    def read_dataframe(data,start='start',end=None,weight=None,use_datetime = False, convert_delta = False):
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
                use_datetime = True
                st = Steps(use_datetime)

                if end is not None:
                    if data[end].dtypes == np.dtype('datetime64[ns]') or use_datetime:         
                        if weight is None:
                            return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None),end = Steps._fill_missing(pd.Timestamp(x[end]),None),use_datetime=use_datetime),axis=1))
                        else:
                            return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None),end = Steps._fill_missing(pd.Timestamp(x[end]),None),weight = x[weight],use_datetime=use_datetime),axis=1))
                    else:
                        raise TypeError("end data must be same type as start data") 
                else:
                    if weight is None:
                        return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None),use_datetime=use_datetime),axis=1))
                    else:
                        return st.add(data.apply(lambda x: Step(start = Steps._fill_missing(pd.Timestamp(x[start]),None),weight=x[weight],use_datetime=use_datetime),axis=1))
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
        else:
            raise TypeError("input data must be a Dataframe")

    @staticmethod
    def read_array(start,end=None,weight=None,use_datetime = False, convert_delta = False):
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


            if use_datetime or Utils.is_date_time(start[0]):
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
        self._cummulative = SortedDict()
        
    def add(self,steps):
        """
        Add an array of individual step objects to this collection of steps.

        Parameters
        ==============
        steps : Array[Steps]. Array of step objects to be added.

        Returns
        ============
        Steps : A new steps object consisting of this object and the additional steps.

        """

        start_steps = [s.detach_child() for s in steps]        
        end_steps = [copy.deepcopy(s.end()) for s in steps if s.end() is not None]

        #keep a copy of the full steps so we can export properly
        self._truesteps = np.append(self._truesteps,steps)

        # the steps operations work best with single step object, no children
        self._steps = np.append(self._steps,start_steps)
        self._steps = np.append(self._steps,end_steps)

        self._truesteps = np.sort(self._truesteps)
        self._steps = np.sort(self._steps)

        #self._using_dt = Utils.is_date_time(self._steps[0]._using_dt)

        self._step_np = np.append(self._step_np,np.array([s.step_np() for s in self._steps]),axis=0)

        self._cummulative = self.to_dict()


        return self

    def __add__(self,other):
        """
        The '+' operation to add int, float, step and steps objects like they are numbers.

        Parameters
        ==============
        other : int, float, step, steps. The thing to add to these steps, a single step or series of steps can be combined with the steps, an single int or float can also
        be added, this will be converted to a single step with a constant basis and added to the steps series.

        Returns
        ============
        Steps : A new steps object consisting of this object with additional step objects representing the other operand.
        
        """
        combine = self.copy()
        if type(other) == Step:
            sc = copy.deepcopy(other)
            sc._using_dt = self._using_dt
            combine.add([sc])
                
            return combine
        
        elif type(other) == Steps:
            combine.add(copy.deepcopy(other.steps()))

            return combine
        else:
            combine.add([Step(use_datetime=self._using_dt,weight=other)])

            return combine

    def __sub__(self,other):
        """
        The '-' operation to subtract int, float, step and steps objects like they are numbers.

        Parameters
        ==============
        other : int, float, step, steps. The thing to subtract from these steps, a single step or series of steps can be combined with the steps, a single int or float can also
        be added, this will be converted to a single step with a constant basis and added to the steps series.

        Returns
        ============
        Steps : A new steps object consisting of this object with additional step objects representing the other operand.
        
        """

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
            combine.add([Step(use_datetime=self._using_dt,weight=-1*other)])

            return combine
    
    def using_datetime(self):
        return self._using_dt
    
    def copy(self):
        return copy.deepcopy(self)
        
    def steps(self):
        return self._steps


    def ecdf(self):
        x = np.sort(self.step_values())
        y = np.arange(0, len(x)) / len(x)

        return x,y

    def ecdf_plot(self,ax=None,**kargs):
        x,y = self.ecdf()

        kind = kargs.pop('kind',None)
        if kind is None:
            kind='bar'
            fill=False

        fill = kargs.pop('fill',None)
        if fill is None:
            fill=False

        edgecolor = kargs.pop('edgecolor',None)
        if edgecolor is None:
            edgecolor=Utils.get_default_plot_color()

        return Utils.simple_plot(x,y,ax=ax,legend=False,kind=kind,fill=fill,**kargs)

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

        return Utils.simple_plot(x,y,ax=ax,legend=False,**kargs)


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


    def rebase(self,new_basis = None,change_steps=False):
        if new_basis is None:
            new_basis = Basis()

        if change_steps:
            self._basis = new_basis
            self._base = new_basis.base()
        else:
            for s in self._steps:
                if s._base is not Bases.constant:
                    s.rebase(new_basis)

    def clip(self,lbound=None,ubound=None):

        data = self.to_dict(False)

        if lbound is None and ubound is None:
            return self
        elif lbound is None:
            new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k <= ubound)])
            
            clip_end = (self.step(ubound))[0]
            new_steps = np.append(new_steps,Step(start=ubound,weight=-1*clip_end))

        elif ubound is None:
            new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k >= lbound)])
            clip_start = (self.step(lbound))[0]
            new_steps = np.append(new_steps,Step(start=lbound,weight=clip_start))

        else:
            if lbound < self._start:
                new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k <= ubound)])
            elif ubound > self._end:
                new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k >= lbound)])
                clip_start = (self.step(lbound))[0]
                new_steps = np.append(new_steps,Step(start=lbound,weight=clip_start))

            else:
                new_steps = np.array([Step(start=k,weight=v) for k,v in data.items() if (v != 0) and (k >= lbound) and (k <= ubound)])

                clip_start = (self.step(lbound))[0]
                clip_end = (self.step(ubound))[0]

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

    def reduce(self, full_reduce=False):      

        data = self.to_dict(False)
            
        new_steps = [Step(start=k,weight=v) for k,v in data.items() if v != 0]
        
        self.clear()
        self.add(new_steps)


    def step_values(self):
        return np.array(self._cummulative.values())
    
    def step_keys(self):
        return list(self._cummulative.keys())
    
    
    def to_dict(self,use_cummulative = True, only_ends = False):
        
        if use_cummulative:
            data:SortedDict = SortedDict()
            
            all_keys = self._step_np[:,0]
            all_values = self._step_np[:,2]*self._step_np[:,3]
            
            # if self._using_dt:
            #     all_keys = np.asarray(list(map(Utils.get_dt, all_keys)))


            # neg_inf_key = Utils.get_epoch_start(self._using_dt)
            # neg_inf_val = self.step(neg_inf_key)[0]

            # if neg_inf_val !=0:
            #     all_values = np.insert(all_values,0,neg_inf_val,axis=0)
            #     all_keys = np.insert(all_keys,0,neg_inf_key,axis=0)

            all_values = np.cumsum(all_values,axis=0)
            # all_values = np.cumsum(all_values,axis=0)

            print(type(all_keys[0]),all_keys[0])
            start_key = np.amin(all_keys)
            print(start_key)

            if start_key == Utils.get_epoch_start(self._using_dt) and len(all_keys) > 1:
                start_key = all_keys[1]
            else:
                start_key = all_keys[0]

            end_key = np.amax(all_keys)
            if end_key == Utils.get_epoch_end(self._using_dt) and len(all_keys) > 2:
                end_key = all_keys[-2]
            else:
                end_key = all_keys[-1]

            #The real value start and end points for the entire series of steps
            self._start = start_key
            self._end = end_key

            #Actually build the dict
            data.update({k:v for k,v in zip(all_keys, all_values)})

            return data
        else:
            data:defaultdict = defaultdict(lambda:0)
            data[Utils.get_epoch_start(self._using_dt)] = self(Utils.get_epoch_start(self._using_dt))[0]

            for s in self._steps:
                data[s.start()] += s.weight()

            return SortedDict(data)
    
    def to_dataframe(self):

        data:array = []

        for s in self._truesteps:
            if s.end() is not None:
                data.append({'start': s.start(),'end':s.end().start(),'value':s.weight()})                  
            else:
                if s._direction == 1:
                    data.append({'start': s.start(),'end':None,'value':s.weight()})
                else:
                    data.append({'start': None,'end':s.start(),'value':s._weight})

        return pd.DataFrame.from_dict(data)


    def __getitem__(self,x):
        return self.step(x)

    def __call__(self,x):
        return self.step(x)
    
    def step(self, x):
        if not hasattr(x,'__iter__'):
            x = [x]
        elif type(x) is slice:
            if Utils.is_date_time(x.start):
                if x.step is None:
                    x = np.arange(x.start,x.stop,timedelta(minutes=1)).astype(pd.Timestamp)
                else:
                    x = np.arange(x.start,x.stop,x.step).astype(pd.Timestamp)
            else:
                x = np.arange(x.start,x.stop,x.step)
        
        

        # bottle neck is right here!
        if len(self._steps) > 0:
            if self._using_dt:
                x = np.asarray(list(map(Utils.get_ts, x)))

            #stvals = np.array([s.step(xf) for s in self._steps])
            #st = np.array([[s._start_ts,s._direction,s._weight] for s in self._steps],dtype=float)
            st = self._step_np[:,[1,2,3]]
            result = np.asarray(fb.fast_steps_heaviside().step(st,x,1))
        else:
            return np.zeros(len(x))

        #result = np.sum(stvals,axis=0)

        del x
        #del stvals

        return result

    def smooth_plot(self,smooth_factor = None, ts_grain = None,ax=None,where='post',**kargs):
        return self.plot(method='smooth',smooth_factor=smooth_factor, ts_grain=ts_grain,ax=ax,where=where,**kargs)

    def plot(self,plot_range=None,method=None, plot_start=None,plot_end=None,
        smooth_factor = None,ts_grain = None,ax=None,where='post',**kargs):

        #hack to correct dt plot slipping
        if self._using_dt:
            dt_delta = pd.Timedelta(hours=11)
        else:
            dt_delta = 0

        if ax is None:
            size = kargs.pop('size',None)
            if size is None:
                size=Utils.get_default_plot_size()

            _, ax = plt.subplots(figsize=size)
        
        color = kargs.pop('color',None)
        if color is None:
            color=Utils.get_default_plot_color()

        if method == None:
            raw_steps = self._cummulative

            #throw off the infinity end points if they are present
            try:
                raw_steps.pop(Utils.get_epoch_start(self._using_dt))
                raw_steps.pop(Utils.get_epoch_end(self._using_dt))
            except:
                pass

            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(minutes=10)
            else:
                ts_grain = 0.00001


            if len(raw_steps.keys()) == 0:
                if plot_range is None:
                    ax.axhline(self(0)[0],color=color, **kargs)
                else:
                    ax.hlines(self(0)[0],plot_range[0],plot_range[1],color=color, **kargs)
            else:
                if plot_range is not None and type(plot_range) in [list,array]:
                    if plot_range[0] != None:
                        start_key = (raw_steps.keys())[0] - plot_range[0]
                        raw_steps[start_key] = self(start_key)
                    else:
                        zero_key = (raw_steps.keys())[0] - ts_grain
                        raw_steps[zero_key] = self(zero_key)
                    
                    if plot_range[1] != None:
                        end_key = (raw_steps.keys())[-1] + plot_range[1]
                        raw_steps[end_key] = self(end_key)
                    else:
                        end_key = (raw_steps.keys())[-1] + ts_grain
                        raw_steps[end_key] = self(end_key)
                else:
                    zero_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[zero_key] = self(zero_key)
                    
                    end_key = (raw_steps.keys())[-1] + ts_grain
                    raw_steps[end_key] = self(end_key)

                    zero_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[zero_key] = self(zero_key)            
                    ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)
                            
                ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)

        elif method == 'pretty':
            raw_steps = self._cummulative

             #throw off the infinity end points if they are present
            try:
                raw_steps.pop(Utils.get_epoch_start(self._using_dt))
                raw_steps.pop(Utils.get_epoch_end(self._using_dt))
            except:
                pass

            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(minutes=10)
            else:
                ts_grain = 0.00001

            if len(raw_steps.keys()) == 0:
                if plot_range is None:
                    ax.axhline(self(0)[0],color=color, **kargs)
                else:
                    ax.hlines(self(0)[0],plot_range[0],plot_range[1],color=color, **kargs)
            else:
                if plot_range is not None and type(plot_range) in [list,array]:
                    if plot_range[0] != None:
                        start_key = (raw_steps.keys())[0] - plot_range[0]
                        raw_steps[start_key] = self(start_key)
                    else:
                        start_key = (raw_steps.keys())[0] - ts_grain
                        raw_steps[start_key] = self(start_key)
                    
                    if plot_range[1] != None:
                        end_key = (raw_steps.keys())[-1] + plot_range[1]
                        raw_steps[end_key] = self(end_key)
                    else:
                        end_key = (raw_steps.keys())[-1] + ts_grain
                        raw_steps[end_key] = self(end_key)
                else:
                    start_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[start_key] = self(start_key)
                    
                    end_key = (raw_steps.keys())[-1] + ts_grain
                    raw_steps[end_key] = self(end_key)

                    start_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[start_key] = self(start_key)            
                    Utils._prettyplot(raw_steps,plot_start=start_key,plot_start_value=0,ax=ax,color=color,**kargs)
                            
                Utils._prettyplot(raw_steps,plot_start=start_key,plot_start_value=0,ax=ax,color=color,**kargs)

            #Steps._prettyplot(raw_steps,plot_start=zero_key,plot_start_value=0,ax=ax,color=color,**kargs)

        elif method == 'function':
                tsx = Utils.get_plot_range(self._start,self._end,ts_grain,use_datetime=self._using_dt)
                ax.step(tsx,self.step(tsx), where=where,color=color, **kargs)
                
        elif method == 'smooth':
            step_ts = np.array([s.start_ts() for s in self._steps if s.start() !=Utils.get_epoch_start(self._using_dt)])
            max_ts = np.amax(step_ts)
            min_ts = np.amin(step_ts)

            if smooth_factor is None:
                smooth_factor = (max_ts - min_ts)/250

            tsx = Utils.get_plot_range(self._start,self._end,ts_grain,use_datetime=self._using_dt)
            ax.plot(tsx,self.smooth_step(tsx,smooth_factor = smooth_factor),color=color, **kargs)

        # elif method == 'experiment':
        #     raw_steps = self._cummulative

        #     # small offset to ensure we plot the initial step transition
        #     if self._using_dt:
        #         ts_grain = pd.Timedelta(minutes=10)
        #     else:
        #         ts_grain = 0.01

        #     zero_key = (raw_steps.keys())[0] - ts_grain
        #     raw_steps[zero_key] = self([zero_key])
        #     ax.step(raw_steps.keys(),self.smooth_step(raw_steps.keys()), where=where,color=color, **kargs)

        else:
            raw_steps = self._cummulative
            
            #throw off the infinity end points if they are present
            try:
                raw_steps.pop(Utils.get_epoch_start(self._using_dt))
                raw_steps.pop(Utils.get_epoch_end(self._using_dt))
            except:
                pass

            # small offset to ensure we plot the initial step transition
            if self._using_dt:
                ts_grain = pd.Timedelta(minutes=10)
            else:
                ts_grain = 0.01

            if len(raw_steps.keys()) == 0:
                if plot_range is None:
                    ax.axhline(self(0)[0],color=color, **kargs)
                else:
                    ax.hlines(self(0)[0],plot_range[0],plot_range[1],color=color, **kargs)
            else:
                if plot_range is not None and type(plot_range) in [list,array]:
                    if plot_range[0] != None:
                        start_key = (raw_steps.keys())[0] - plot_range[0]
                        raw_steps[start_key] = self(start_key)
                    else:
                        zero_key = (raw_steps.keys())[0] - ts_grain
                        raw_steps[zero_key] = self(zero_key)
                    
                    if plot_range[1] != None:
                        end_key = (raw_steps.keys())[-1] + plot_range[1]
                        raw_steps[end_key] = self(end_key)
                    else:
                        end_key = (raw_steps.keys())[-1] + ts_grain
                        raw_steps[end_key] = self(end_key)
                else:
                    zero_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[zero_key] = self(zero_key)
                    
                    end_key = (raw_steps.keys())[-1] + ts_grain
                    raw_steps[end_key] = self(end_key)

                    zero_key = (raw_steps.keys())[0] - ts_grain
                    raw_steps[zero_key] = self(zero_key)            
                    ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)
                            
                ax.step(raw_steps.keys(),raw_steps.values(), where=where,color=color, **kargs)

        return ax

    def _operate_norm(self,other, op_func):
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
            
            all_keys = [s.start() for s in self._steps]
            #all_keys.append(Utils.get_epoch_start(self._using_dt))
            all_values = self.step(all_keys)

            mask = np.where(op_func(all_values,other), np.sign(all_values),0)

            groups = [(group[0],group[-1]) for group in (list(group) for key, group in groupby(range(len(mask)), key=mask.__getitem__) if key!=0)]

            for g in groups:
                s = g[0]
                e = g[-1]

                if s == 0 and e >= len(self._steps)-1:
                    return new_instance.add([Step(start=self._steps[0].start(),end=self._steps[-1].start(),weight=np.sign(all_values[s]))])

                if s != e:
                    e = g[-1] if e >= len(self._steps)-1 else g[-1]+1
                    new_steps.append(Step(start=self._steps[s].start(), end=self._steps[e].start(),weight=np.sign(all_values[s])))
                else:
                    if s+1 >=len(self._steps):
                        new_steps.append(Step(start=self._steps[s].start(),weight=np.sign(all_values[s])))
                    else:
                        new_steps.append(Step(start=self._steps[s].start(), end=self._steps[s+1].start(),weight=np.sign(all_values[s])))
                
            new_instance.add(new_steps)
            return new_instance


    def _operate_norm_old(self,other, op_func):
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
            
            #all_keys = [s.start() for s in self._steps]
            #all_values = self.step(all_keys)

            #all_keys = np.array(list(self._cummulative.keys()))
            #all_values = np.array(list(self._cummulative.values()))

            all_keys = [s.start() for s in self._steps]
            all_values = self.step(all_keys)

            mask = np.where(op_func(all_values,other), True,False)
            
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
                if self(Utils.get_epoch_start(self._using_dt)) == 0:
                    if self(Utils.get_epoch_end(self._using_dt)) != 0:
                        return new_instance.add([Step(start=self._start)])
                    else:
                        return new_instance.add([Step(start=self._start,end=self._end)])
                else:
                    if self(Utils.get_epoch_end(self._using_dt)) != 0:
                        return new_instance.add([Step(use_datetime=self._using_dt)])
                    else:
                        return new_instance.add([Step(end=self._end)])
                    
            new_instance.add(new_steps)
            return new_instance

    def _operate_value_new(self,other, op_func):
        """
        This function is used to create filtered version of the steps by removing steps not evaluating to true from applying the comparison function
        to the cummulative total of the steps.

        Parameters
        ===========
        other : int, float, Step, Steps. Any value to compare each step component against.
        op_func : Numpy Universal Function. A binary comparison function that returns a bool, e.g >,<,==.

        Returns:
        ==========
        Steps: A new steps instance with only step locations everywhere the filter condition was true.
        
        """

        if type(other) in [float,int]:
            new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)

            new_steps = np.array([],dtype=Step)
            
            all_keys = [s.start() for s in self._steps]
            all_values = self.step(all_keys)

            mask = np.where(op_func(all_values,other), True,False)

            groups = [(group[0],group[-1]) for group in (list(group) for key, group in groupby(range(len(mask)), key=mask.__getitem__) if key)]

            for g in groups:
                s = g[0]
                e = g[-1]

                if s == 0 and e >= len(self._steps)-1:
                    return self

                if s != e:
                    e = g[-1] if e >= len(self._steps)-1 else g[-1]+1
                    #add new first and last steps
                    first_step = self._steps[s].copy()
                    first_step._weight = all_values[s]
                    adjustment = np.sum([s.weight() for s in self._steps[s+1:e]])
                    last_step = self._steps[e].copy()
                    last_step._weight = -1*all_values[s] + adjustment

                    new_steps = np.append(new_steps,[first_step,last_step])
                    new_steps = np.append(new_steps,copy.deepcopy(self._steps[s+1:e]))

                else:
                    first_step = self._steps[s].copy()
                    first_step._weight = all_values[s]
                    last_step = self._steps[s+1].copy()
                    last_step._weight = -1*all_values[s]

                    new_steps = np.append(new_steps,[first_step,last_step])
                    #new_steps.append(self._steps[s+1:e-1])

                    #new_steps.append(Step(start=self._steps[s].start(), end=self._steps[s+1].start(),weight=np.sign(all_values[s])))
                
            new_instance.add(new_steps)
            return new_instance

    def _operate_value(self,other, op_func):
        """
        This function is used to create filtered version of the steps by removing steps not evaluating to true from applying the comparison function
        to the cummulative total of the steps.

        Parameters
        ===========
        other : int, float, Step, Steps. Any value to compare each step component against.
        op_func : Numpy Universal Function. A binary comparison function that returns a bool, e.g >,<,==.

        Returns:
        ==========
        Steps: A new steps instance with only step locations everywhere the filter condition was true.
        """

        if type(other) in [float,int]:
            new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
            new_steps = []
            #all_keys = np.array(list(self._cummulative.keys()))
            #all_values = np.array(list(self._cummulative.values()))
            
            all_keys = [s.start() for s in self._steps]
            all_values = self.step(all_keys)

            mask = np.where(op_func(all_values,other), True,False)
            
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
                    elif not (st is None) and (s.start_ts() > st.start_ts()):
                        new_steps.append(Step(start=s.start(),weight=s.weight()))
                        adj += s.weight()
                else:
                    all_true = False
                    if not first:
                        first=True
                        new_steps.append(Step(start=s.start(),weight=-1*(self._cummulative[st.start()] + adj)))
                        adj = 0
                        st = None
            
            if all_true:
                return self

            new_instance.add(new_steps)

            return new_instance

    def reflect(self,reflect_point = 0):
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        
        reflected_steps = [s.reflect(reflect_point) for s in self._steps]
        new_instance.add(reflected_steps)
        
        return new_instance
    
    def __pow__(self,power_val):
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        
        pow_steps = [s**power_val for s in self._steps]
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

    def normalise(self):
        return self._operate_norm(0, operator.ne)
    
    def invert(self):
        return self._operate_norm(0, operator.eq)
        
    def __gt__(self,other):
        return self._operate_value(other, operator.gt)
    
    def __lt__(self,other):
        return self._operate_value(other, operator.lt)

    def __ge__(self,other):
        return self._operate_value(other, operator.ge)
    
    def __le__(self,other):
        return self._operate_value(other, operator.le)

    def __ne__(self,other):
        return self._operate_value(other, operator.ne)

    def __eq__(self,other):
        return self._operate_value(other, operator.eq)
        
    def __lshift__(self,other):
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        
        lshift_steps = [s<<other for s in self._steps]
        new_instance.add(lshift_steps)
        return new_instance
        
    def __rshift__(self,other):
        new_instance = Steps(use_datetime=self._using_dt,basis=self._basis)
        
        rshift_steps = [s>>other for s in self._steps]
        new_instance.add(rshift_steps)
        return new_instance

    def __floordiv__(self,other):
        pass

    def rotate(self):
        return Steps.read_array(self.step_values(),self.step_keys(),convert_delta=True)


    def __truediv__(self,other):
        return self*other**-1

    def __mul__(self,other):
        if isinstance(other, Steps):
            new_steps = np.array([],dtype=Step)

            #Need to remove the inserted end steps as the start step parent will handle the end in the Step multiplication
            #end_steps = [s.end() for s in self._steps if s.end() is not None]
            other_end_steps = [s.end() for s in other.steps() if s.end() is not None]

            for s in self._steps:
                #if s not in end_steps:
                for s_other in other.steps():
                    #if s_other not in other_end_steps:
                    new_steps = np.append(new_steps,s*s_other)

            st = Steps().add(new_steps)
            st.reduce()

            return st

        else:
            new_steps = np.array([],dtype=Step)

            #Need to remove the inserted end steps as the start step parent will handle the end in the Step multiplication
            end_steps = [s.end() for s in self._steps if s.end() is not None]

            for s in self._steps:
                #if s not in end_steps:
                new_steps = np.append(new_steps,s*other)

            st = Steps().add(new_steps)
            st.reduce()

            return st

    def start_ts(self):
        pass
    
    def start(self):
        pass
    
    def end(self):
        pass
    
    def weight(self):
        pass

    def __repr__(self):
        return ','.join([str(s) for s in self._steps])

    def link_child(self, other):
        pass
    
    def __irshift__(self,other):
        pass
    
    def __ilshift__(self,other):
        pass
    
    def smooth_step(self,x,smooth_factor:Union[int,float] = None,smooth_basis:Basis = None):

        step_ts = np.array([s.start_ts() for s in self._steps if s.start() != Utils.get_epoch_start(self._using_dt)])
        max_ts = np.amax(step_ts)
        min_ts = np.amin(step_ts)

        if smooth_factor is None:
            smooth_factor = (max_ts - min_ts)/250

        if smooth_basis is None:
            smooth_basis = Basis(Bases.logit(),smooth_factor)
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
    
    def std(self):
        return np.sqrt(Analysis.var(self))

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
    
    def pacf(self, maxlags = None):
        #l = len(self._cumsum)
        l = len(self.step_values())

        if (maxlags is None) or (maxlags >= l):
            maxlags = int(0.1*l) 

        return Analysis.pacf(self, maxlags)

    def pacf_step(self, maxlags = None):
        lags, pac = self.pacf(maxlags)

        return Steps.read_array(start=lags, weight=pac, convert_delta=True)

    def pacf_plot(self, maxlags = None,ax=None,**kargs):
        lags, pac = self.pacf(maxlags)

        kind = kargs.pop('kind',None)
        if kind is None:
            kind='bar'

        return Utils.simple_plot(lags,pac,ax=ax,legend=False,kind=kind,**kargs)

    def percentile(self,percent):
        return Analysis.percentile(self,percent)

    def covariance(self,other):
        return Analysis.covariance(self,other)

    def correlation(self,other):
        return Analysis.correlation(self,other)