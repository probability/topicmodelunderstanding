from __future__ import annotations
import numpy as np

from typing import Union
from hotstepper.Bases import Bases

class Basis():

    def __init__(self,bfunc=None, param:float=1,lbound:float= -np.Inf,ubound:float = np.Inf) -> None:
        self.lbound = lbound
        self.ubound = ubound
        self.param = param
        
        if bfunc is None:
            self._base = Bases.heaviside()
        else:
            self._base = bfunc
            
    def base(self):
        return self._base