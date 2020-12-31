from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime
import pandas as pd
from functools import partial
from typing import Optional, Union
import scipy.stats as st
from datetime import datetime, timedelta
from numba import jit, prange