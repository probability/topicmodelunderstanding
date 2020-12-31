# HotStepper
The HotStepper library is for exploring datasets via step function expansions.

What the f*&^ is a step function you may ask?, Have you ever seen some stairs?, this is an equal oppotunity package, so you may be in a wheel chair and unable to use stairs in the typical way, so just seeing some stair will suffix.

Instead of providing a strict methatical definition that we can all wank off about, how bout just some simple comparisons to warm us up? If you still need to have a wank, feel free to step out (pun intended) anytime.

What is a function? ok, how about just some data we could plot? let's go home school, say we have a vector...oh f&^%, what is that? ok ok, how about just a list of numbers, say y = (2,3,5,8,13,21), to keep the wanking impulse alive, we could say that this is a discrete function where we can index the values from left to right with an integer, for example <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ay%28x%29+%3D+%282%2C3%2C5%2C8%2C13%2C21%29%2C+%7Bx%3A+x%5Cin+%5Cmathbb%7BN%7D%7D%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
y(x) = (2,3,5,8,13,21), {x: x\in \mathbb{N}}
\end{align*}
">, so that we could do something fancy like y(3) = 8, since we are starting at n = 0.

Alright, if we just plot y(n) with straight lines connecting the points, we'd get something like,

```python
    x = np.arange(0,6,1,dtype=int)
    y = np.array([2,3,5,8,13,21],dtype=int)

    plt.plot(x,y)
```
<p align="left"><img src="https://github.com/TangleSpace/HotStepper/blob/main/docs/images/fibo_line.png?raw=true" title="Fibonacci Plot" alt="Fibonacci Plot"></p>

Or we could get fancy and use step functions to construct the same plot from the rules for fibonacci sequence.

```python
    x = np.arange(0,6,1,dtype=int)
    y = np.array([2,3,5,8,13,21],dtype=int)
    fibo_deltas = np.diff(y,prepend=0)

    st = Steps().add([Step(i,None,fn) for i, fn in enumerate(fibo_deltas)])
    st.plot()
```
<p align="left"><img src="https://github.com/TangleSpace/HotStepper/blob/main/docs/images/fibo_steps.png?raw=true" title="Fibonacci Step Plot" alt="Fibonacci Step Plot"></p>

