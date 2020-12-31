# HotStepper
The HotStepper library is for exploring datasets via step function expansions.

What the f*&^ is a step function you may ask?, Have you ever seen some stairs?, this is an equal oppotunity package, so you may be in a wheel chair and unable to use stairs in the typical way, so just seeing some stair will suffix.

Instead of providing a strict methatical definition that we can all wank off about, how bout just some simple comparisons to warm us up? If you still need to have a wank, feel free to step out (pun intended) anytime.

What is a function? ok, how about just some data we could plot? let's go home school, say we have a vector...oh f&^%, what is that? ok ok, how about just a list of numbers, say y = (2,3,5,8,13,21), to keep the wanking impulse alive, we could say that this is a discrete function where we can index the values from left to right with an integer, for example y(n) = (2,3,5,8,13,21), Domain: {n : n in Nautral Numbers}, so that we could do something fancy like y(3) = 8, since we are starting at n = 0.

Alright, if we just plot y(n) with straight lines connecting the points, we'd get something like,
We can find the average number of viewers, per hour of the day, and plot:

```python
>>> Steps().add([Step(2),Step(1),Step(2),Step(3),Step(5),Step(8)])
```
<p align="left"><img src="https://github.com/TangleSpace/HotStepper/blob/main/docs/images/fibo_steps.png?raw=true" title="Fibonacci Steps" alt="Fibonacci Steps"></p>

