from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import abc
import pendulum as pdate
from sortedcontainers import SortedDict
from datetime import datetime
from typing import Union


valid_input_types = (int,float,pd.Timestamp,datetime)

class Utils(metaclass=abc.ABCMeta):
    
    @staticmethod
    def input_types():
        return valid_input_types

    @staticmethod
    def get_default_plot_color():
        return '#9c00ff'

    @staticmethod
    def get_default_plot_size():
        return (16,8)


    @staticmethod
    def get_epoch_start(use_datetime = True):
        if use_datetime:
            return pd.Timestamp(1999,12,31,23,59)
        else:
            return -np.inf

    @staticmethod
    def get_epoch_end(use_datetime = True):
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
    def _modify_step(obj, attr,new_value, change_end = False):
        if obj is not None and hasattr(obj,attr):
            setattr(obj,attr,new_value)

        if change_end and obj._end is not None and hasattr(obj._end,attr):
            setattr(obj,attr,new_value)

    @staticmethod
    def get_keys(val, is_dt = False, is_inf = False):

        if (is_inf or val is None) or (val < Utils.get_epoch_start(is_dt)):
            val = Utils.get_epoch_start(is_dt)

        return val, Utils.get_value(val,is_dt)

    @staticmethod
    def simple_plot(xdata,ydata,cdata=None, ax=None,**kargs):
        if ax is None:
            _, ax = plt.subplots()

        dfplot = pd.DataFrame()
        dfplot['x'] = xdata
        dfplot['y'] = ydata

        color = kargs.pop('color',None)
        if color is None:
            color=Utils.get_default_plot_color()
        
        if cdata is None:
            dfplot.plot(x='x',y='y', ax=ax,color=color, **kargs)
        else:
            dfplot['c'] = cdata
            dfplot.plot(x='x',y='y', c='c', ax=ax,color=color, **kargs)

        return ax

    @staticmethod
    def _prettyplot(step_dict,plot_start=0,plot_start_value=0,ax=None,start_index=1,end_index=None,include_end=True,**kargs):

        step0_k = plot_start
        step0_v = plot_start_value

        color = kargs.pop('color',None)
        if color is None:
            color=Utils.get_default_plot_color()

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
    def get_plot_range(start,end, delta = None, use_datetime = False):

        shift = None

        if Utils.is_date_time(start) or use_datetime:
            start = pd.to_datetime(start)
            if end is not None:
                end = pd.to_datetime(end)
                shift = end - start
                span_seconds = shift.value
                shift = pd.Timedelta(nanoseconds=int(0.05*span_seconds))

                if delta is None:
                    if int(0.01*span_seconds) > 0:
                        delta = pd.Timedelta(nanoseconds=int(0.005*span_seconds))
                    else:
                        delta = pd.Timedelta(seconds=10)
                
                return np.arange(start-shift, end + shift, delta).astype(pd.Timestamp)
            else:
                if start.hour > 0:
                    shift = pd.Timedelta(hours=5)
                    delta = pd.Timedelta(hours=1)
                elif start.minute > 0:
                    shift = pd.Timedelta(minutes=6)
                    delta = pd.Timedelta(minutes=2)
                else:
                    shift = pd.Timedelta(hours=18)
                    delta = pd.Timedelta(hours=3)

                return np.arange(start-shift, start + shift, delta).astype(pd.Timestamp)
        else:
            if end is not None:
                shift = 0.03*(end - start)

                if delta is None:
                    delta = 0.1*shift

                return np.arange(start-shift, end + shift, delta)
            else:
                shift = 0.1*start

                if delta is None:
                    delta = 0.1*shift

                return np.arange(start-shift, start + shift, delta)

    @staticmethod
    def is_date_time(value):
        return hasattr(value,'timestamp') and callable(value.timestamp)

    @staticmethod
    def get_ts(ts):
        if Utils.is_date_time(ts):
            return ts.timestamp()
        else:
            return ts

    @staticmethod
    def get_dt(ts):
        return pd.Timestamp.utcfromtimestamp(int(ts))
   