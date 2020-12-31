# HotStepper
The HotStepper library is for exploring datasets via step function expansions.

What the f*&^ is a step function you may ask?, Have you ever seen some stairs?, this is an equal oppotunity package, so you may be in a wheel chair and unable to use stairs in the typical way, so just seeing some stair will suffix.

Instead of providing a strict methatical definition that we can all wank off about, how bout just some simple comparisons to warm us up? If you still need to have a wank, feel free to step out (pun intended) anytime.

What is a function? ok, how about just some data we could plot? let's go home school, say we have a vector...oh f&^%, what is that? ok ok, how about just a list of numbers, say y = (2,3,5,8,13,21), to keep the wanking impulse alive, we could say that this is a discrete function where we can index the values from left to right with an integer, for example $y(n) = (2,3,5,8,13,21), Domain: {n : n \in \mathbb{N}}$, so that we could do something fancy like y(3) = 8, since we are starting at n = 0.

# GFM url syntax
# https://render.githubusercontent.com/render/math?math={math_equation}
# equation syntax
F=P(1%2B\frac{i}{n})^{nt}            # normal, valid
\large F=P(1%2B\frac{i}{n})^{nt}     # large: before encoding spaces, invalid
%5Clarge%20F=P(1%2B\frac{i}{n})^{nt} # large: after `%20` space and `%5C` backslash/reverse solidus `\` encoding, valid
\large+F=P(1%2B\frac{i}{n})^{nt}     # large: after `+` space encoding, valid
# e.g. https://render.githubusercontent.com/render/math?math=e%5E%7Bi%5Cpi%7D%20%3D%20-1
#
# CodeCogs url syntax
# https://latex.codecogs.com/svg.latex?{math_equation}
# equation syntax
F=P(1+\frac{i}{n})^{nt}            # normal, valid
\large F=P(1+\frac{i}{n})^{nt}     # large: before encoding spaces, invalid
%5Clarge%20=P(1+\frac{i}{n})^{nt}  # large: after `%20` space and `%5C` backslash/reverse solidus `\` encoding, valid
# e.g. https://latex.codecogs.com/svg.latex?%5Clarge%20F=P(1+\frac{i}{n})^{nt}


Alright, if we just plot y(n) with straight lines connecting the points, we'd get something like,
We can find the average number of viewers, per hour of the day, and plot:

```python
>>> pd.Series([views.mean(60*i, 60*(i+1)) for i in range(24)]).plot()
```
<p align="left"><img src="https://github.com/venaturum/staircase/blob/master/docs/img/meanperhour.png?raw=true" title="mean page views per hour" alt="mean page views per hour"></p>


"# HotStepper" 
