import sys
sys.path.insert(0, r".\\")

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hotstepper.Basis import Basis
from hotstepper.Step import Step
from hotstepper.Steps import Steps
from hotstepper.Steps import Analysis


def test_pow():
    s1 = Step(start=10,end=15,weight=-2)
    s2 = Step(start=12,weight=-1)
    s3 = Step(end=13,weight=2.5)
    s3n = Step(end=13,weight=-2.5)
    s4 = Step(start=14,end=16.5)
    s5 = Step(start=15,weight=2)
    s6 = Step(start=16,weight=2)
    s6n = Step(start=16,weight=-2)
    s7 = Step(start=13.5,end=14.5,weight=2)

    #compare direct and shortcut methods
    assert (s1**2) == (s1*s1)
    assert (s3**2) == (s3*s3)
    assert (s3n**2) == (s3n*s3n)
    assert (s5**2) == (s5*s5)
    assert (s7**2) == (s7*s7)
    assert (s6**2) == (s6*s6)
    assert (s6n**2) == (s6n*s6n)

    assert (s1**3) == (s1*s1*s1)
    assert (s3n**3) == (s3n*s3n*s3n)
    assert (s3**3) == (s3*s3*s3)
    assert (s5**3) == (s5*s5*s5)
    assert (s7**3) == (s7*s7*s7)
    assert (s6**3) == (s6*s6*s6)
    assert (s6n**3) == (s6n*s6n*s6n)


def test_reflect():
    s1 = Step(start=10,end=15,weight=-2)
    s2 = Step(start=12,weight=-1)
    s3 = Step(end=13,weight=2.5)
    s3n = Step(end=13,weight=-2.5)
    s4 = Step(start=14,end=16.5)
    s5 = Step(start=15,weight=2)
    s6 = Step(start=16,weight=2)
    s6n = Step(start=16,weight=-2)
    s7 = Step(start=13.5,end=14.5,weight=2)

    #compare direct and shortcut methods
    assert s1.reflect() == s1*-1
    assert s3n.reflect() == s3n*-1
    assert s3.reflect() == s3*-1
    assert s5.reflect() == s5*-1
    assert s7.reflect() == s7*-1
    assert s6.reflect() == s6*-1
    assert s6n.reflect() == s6n*-1
