from imports import *
import staircase as sc
import operator
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r"../data/vessel_queue.csv", parse_dates=['enter', 'leave'], dayfirst=True)

vsteps = Steps.read_dataframe(df,'enter','leave')