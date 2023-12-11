#### Part A: Stochastic Computing [2 pt/question, 8 points]

The provided scaffold code runs 10,000 trials of the `0.8*0.4 + 0.6` stochastic computation and plots the distribution of results. The `PART_A_example_computation` function executes the above numeric expression, and the `execute_stochastic_computation` runs a stochastic computation for <ntrials> trials and plots the distribution + reports the mean and standard deviation of the stochastic executions. The vertical red line shows the expected result. Currently, the stochastic computing operations and value-to-bitstream / bitstream-to-value conversion routines are not implemented. 

Implement the stochastic computing paradigm by filling in the function stubs in the PosStochComputing class. don't worry about supporting negative numbers just yet.

- `to_stoch` : convert a value in [0,1] to a stochastic bitstream.
- `from_stoch`: convert a stochastic bitstream to a numerical value
- `stoch_add`: perform scaled addition over two stochastic bitstreams.
- `stoch_mult`: perform multiplication over two stochastic bitstreams.

Q1. How does the mean change with increasing bitstream length? How does the variance change?

__A1:__ As we increase the length of the bitstream the mean gets of several trials get closer to the reference expected value of 0.46.
The variance also decreases as we increase the length of the bitstream.
The results are shown in the following table:

| # of trials | mean     | variance |
|-------------|----------|----------|
| 10          | 0.458310 | 0.158660 |
| 100         | 0.458904 | 0.049456 |
| 1000        | 0.459801 | 0.015723 |

---

Q2. What is the smallest representable numeric value in 1000-bit stochastic bitstream? What happens when you try to generate a bitstream for this value -- do the bitstream values converge to the desired value?

__A2:__ The smallest representable (non-zero) value would be 1/1000 or 0.001.
When generating many bitstreams for the value and parsing them back as values, we see that while 0.001 is the mode of the output distribution, the values of 0 and 0.002 are also very common, causing a large relative error.
While the absolute standard deviation of this value is technically lower than for instance 0.5, the relative error is much higher.
This could cause issues when multiplying small values.

---

Q3. Using what you learned from the analysis in (Q2), design a stochastic computation that produces an incorrect result for bitstreams with a length n=1000. You must accomplish this with stochastic operations, all values must be >= 0.1 and every constant must be a uniquely generated bitstream. Is it possible to fix this issue?

__A3:__ The mean of a bitstream can only represent numbers between 0 and 1 in increments of 1/n, so if we choose a computation that results in a number that is not a multiple of 1/n, we will always get an incorrect result.
For n = 1000, that means the number should have more than 3 decimals after the decimal point.
We chose to perform 0.353 * 0.5 = 0.1765, which is not a multiple of 0.001.
Testing this in the code, we see that the output never matches the reference value.

---

Q4. What stochastic bitstream length L do we need to represent a value V, assuming V in [0,1] and V != 0? Write the equation.

__A4:__ Assuming V in this question refers to the smallest representable value, the equation would be 1/V.
For other values, the smallest representable difference would be .

If we want just to be able to represent V, we need to find an integer L such that 1/L divides V.
This means that V = 1/L * N for some N, where N is an integer.
If we fix V, then this means we look for an integer L such that L = N / V for some integer N.
The smallest such L is the required bitstream length to represent V.

If V is a decimal with D decimal places, then we know that V can be written as V = K / 10^D for some integer K.
Thus if we choose N = K, we get that L = N / V = K / (K / 10^D) = 10^D.
So then L is upper bounded by 10^D, L <= 10^D.

---


#### Part X: Non-Idealities Stochastic Computing [2 pt/question, 4 points]

Next, we'll experiment with introducing non-idealities into the stochastic computation. We will introduce two kinds of non-idealities:

- bit-shift errors: this class of errors result from timing issues in circuits. This non-ideality causes one bitstream to lag behind the other during computation. For example, a bit shift error at index 1 of a bit stream would transform some stream 00101 to 00010.


- bit-flip errors: this class of errors results from bit flips in storage elements, or in the computational circuit. This non-ideality introduces bit flips in bitstreams. For example, a bit flip error at index 1 of a bit stream would transform some stream 00101 to 01101.

Fill in the `apply_bitshift` and `apply_bitflip` functions in the stochastic computing class and apply these non-idealities at the appropriate points in the stochastic computing model. Make sure these non-idealities are only enabled for this section of the homework.

Q1. What happens to the computational results when you introduce a per-bit bit-flip error probability of 0.0001? What happens when the per-bit bit flip error probability is 0.01?

__A1:__ Applying a per-bit bit-flip error rate of 0.0001 has no observed effect on the output at bitstream length 1000.
When increasing the probability to 0.01, the mean of the modified example computation slightly increases (from 0.459980 to 0.461141), and this effect is consistent across many runs.
The variance stays roughly the same, as although the output number is being pushed closer to 0.5, the noise introduced by random bit flips contributes to the variance.
The reason for this is that when random bits are flipped at a relatively high rate, streams with more 1s will have a higher likelihood of having 1s flip to 0s than vice-versa.
This means the mean will increase, with the variance of this number of several runs decreasing, as it is being pushed closer to 0.5.
For numbers with more 0s than 1s (lower mean than 0.5), they will be shifted up due to the random bit-flipping.

---

Q2. What happens to the computational  results when you introduce a per-bit bitshift error probability of 0.0001? What happens when the per-bit bit shift error probability is 0.01?

__A2:__ The per-bit bit-shift error rate of 0.0001 also has no visible effect on the output bitstream.
Increasing the probability to 0.01, we see the mean of the output values significantly decrease from 0.459980 to 0.448127, an effect that can be explained by each shift having a chance to push out a 1 on the right, while always introducing a 0 on the left.
This means that the number of 1s in the bitstream will decrease in expectation, while the number of 0s increase, pushing the mean down.
If the shifting operation had been defined by rotation instead of shifting (moving the rightmost bit around to the left), the mean value would not change.

---

Q3. In summary, is the computation affected by these non-idealities? Do you see any changes in behavior as the bitstream grows?

__A3:__ The bitstream is minimally affected by bit-flip errors, but can be effected by bit-shift errors, depending on how they are realized in practice.
As we grow the size of the bitstream, these effects become more pronounced in some individual cases, but in expectation they stay the same.
With a longer bitstream we can more easily predict the effect of the non-idealities on each individual bitstream, but the distributional effects stay the same.
This is in part because of how we have modeled the non-idealities, as they are applied to each bit, and the number of bits grows with the length of the bitstream.

---


#### Part Y: Statically Analyzing Stochastic Computations [2 pt/question, 8 points]

Next, we'll build a simply static analysis for stochastic computations. A _static analysis_ is a type of analysis that is able to infer information about a program without ever running the computation. The analysis we will be building determines the minimum bitstream size necessary for a computation, given a set of precisions for each of the arguments. For example, to compute the bitstream length for the following expression:

    (x + y) + z

We will invoke the following set of functions:

    `get_size(stoch_add(stoch_add(prec_x,prec_y), prec_z))`

If the precision of x is 0.01, the precision of y is 0.02 and the precision of z is 0.03, then the minimum bitstream length is 100. In this exercise, you will be populating the `StochasticComputingStaticAnalysis` class, which offers the following functions:

    - `stoch_var`, given a variable with a desired precision `prec`, update the static analyzer to incorporate this informatin.
    - `stoch_add`, given two stochastic bitstreams that can represent values with precision `prec1` and `prec2` respectively, figure out the precision required for the result stochastic bitstream given an addition operation is performed. Update the static analyzer to incorporate any new information.
    - `stoch_mult`, given two stochastic bitstreams that can represent values with precision `prec1` and `prec2` respectively, figure out the precision required for the result stochastic bitstream given a multiplication operation is performed. Update the static analyzer to incorporate any new information.
    - `get_size`, given all of the operations and variables analyzed so far, return the smallest possible bitstream size that accurately executes all operations, and can accurately represent all values.


We will use this static analysis to figure out what stochastic bistream length to use for the computation (w*x + b), where the smallest value of w is 0.01, the smallest value of x is 0.1, and the smallest value of b is 0.1. For convenience, the scaffold file provides helper functions `PART_Y_analyze_wxb_function` for analyzing the `w*x+b` function, given a dictionary of precisions for variables `w`, `x`, and `b`, a `PART_Y_execute_wxb_function` which executes the `w*x+b` function using stochastic computing given a dictionary of variable values for `w`, `x`, and `b`, and a `PART_Y_test_analysis` function which uses the static analysis to find the best bitstream size for the `w*x+b` expresison, and then uses the size returned by the static analyzer to execute the `w*x+b` for ten random variable values that have the promised precisions.

Q1. Describe how your precision analysis works. 

__A1:__ I think the term 'precision' seems to mean both the number of bits in the required bitstream and the minimum representable value at different points in the text/code.
I will use the term as used in the code (although some comments still contradict this), where the precision of a bitstream is the minimum representable value.
We compute the minimum precision required to represent the result of each operation as the minimum possible result of the operation.
For addition, this is the sum of the minimum possible values of the operands.
For multiplication, this is the product of the minimum possible values of the operands.
We log the precision required for a minimum value of each variable in the computational graph in a list.
The precision required to perform the entire computation (including representing the partial results) is then the minimum required precision of all the variables in the graph.
To get the number of bits required in this case, we take the ceil(1/precision) for the minumum precision.

---

Q2. What bitstream length did your analysis return?

__A2:__
The analysis of the example computation given for part Y is <u>1000 bits</u>.

---

Q3. How did the random executions perform when parametrized with the analyzer-selected bitstream length?

__A3:__
In each set of trials we get a mean that is very close to the reference result, with a standard deviation of around 0.014.
This is acceptable performance.

---

Q4. What if you execute the computation with values w=0.00012, x = 0.124, and b = 0.1? Would you expect the result to be accurate? Why or why not?

__A4:__
If we break the rules used for the analysis and use values that are below the specified minimums, we get a high relative standard deviation of over 0.1.
In expectation the result is correct, and we can in fact represent numbers at the scale of the final result of the computation.
However, the partial result of w * x should be 0.0000149, a number that is below the minimum representable value in our computation with a bitstream length of 1000 (0.001).
Running the analysis on this set of parameters shows that we should have used a bitstream length of at least 83334 for this computation.
This experiment shows surprising robustness to even unrepresentable values, since repeating the experiment many times can give us a correct result.
 
#### Part Z: Sources of Error in Stochastic Computing [2 points + 2 points extra credit]

Next, we will investigate the `PART_Z_execute_rng_efficient_computation` stochastic computation. This computation implements x*x+x, and implements an optimization (`save_rngs=True`) that reuses the bitstream for x to reduce the number of random number generators.

Q1. Does the accuracy of the computation change when the `save_rngs` optimization is enabled? Why or why not?

__A1:__
The accuracy of the computation changes wildly with the `save_rngs` 'optimization' enabled.
The reason for this is that the bits are perfectly correlated, which means each addition produces the same result as the original value (since we are doing weighted sampling of the same bitstream!).
For an input of x = 0.3 we expect the reference value of 1.95, but we instead get a mean result of 0.3 out.
This is because computing x * x with perfectly correlated bitstreams always results in the same value x (with the same bits set), and adding this to the original value of x will randomly sample bits from either x or x, which obviously also returns x.
So the result of the computation (x * x + x) is incorrectly computed as x.

Q2. Devise an alternate method for implementing $x*x+x$ from a single stochastic bitstream. There is a way to do this with a single N+k-bit bitstream, where k is a small constant value. What did you do?

__A2:__ 
We can do this even without the k extra bits, if we allow for the bitstream to be rotated, although this requires access to the entire bitstream, and won't work with a streaming implementation.
However, in the case where we are streaming, a possible approach is to add k = 2 extra bits to the bitstream.
Then we stagger the second use of x by 1 and the third use of x by 2, thus shifting all the bits by 1 and 2 places respectively.
This means that each interaction will have uncorrelated bits interacting with each other, since each operation happens bitwise between the bitstreams.
If we needed to use the same bitstream L times, we would need k = L - 1 extra bits to be able to do this.
Implementing the approach shows that we get the correct answer!

---
 
#### Part W: Extend the Stochastic Computing Paradigm [15 points]

Come up with your own extension, application, or analysis tool for the stochastic computing paradigm. This is your chance to be creative. Describe what task you chose, how it was implemented, and describe any interesting results that were observed. Here are some ideas to get you started:

- Implement a variant of stochastic computing, such as deterministic stochastic computing or bipolar stochastic computing. You may also modify the existing stochastic computing paradigm to incorporate a new source of hardware error -- you will need to justify your hardware error model. 

- Build a stochastic computing analysis of your choosing. You may build up the existing bitstream size analysis to work with abstract syntax trees, or you may devise a new analysis that studies some other property of the computation, such as error propagation or correlation.

- Implement an application using the stochastic computing paradigm. Examples from literature include image processing, ML inference, and LDPC decoding.

__A:__
I implemented a tracing analysis for stochastic computing that works with arbitrary python functions that contain the operations we have learned about in this class.
The analysis works by parsing the function to an AST to understand what variables are inputted, then passing tracer objects into the function and evaluating it on them.
Each tracer object creates a new tracer when transformed in an operation like an assign, addition or multiplication, and we allow for floating points number to be multiplied in with the tracers.
The result is a class that can be used for static analysis (`StaticAnalysisAST` with `TracingPrecision`).

Another function allows for the stochastic computing to be built by tracing arbitrary functions of supported operators.
This works by tracing inputted variables and wrapping any interacting value in a tracer object.
The return value from the function we trace will be a tracer object that can be used to run the stochastic computation by calling `StochasticComputeTracing.doit(self)`.
When this is run, the tracer object will recursively call the function, sampling the bitstreams at the appropriate points.
Because of how the method `doit` is set up, the function will never use the same bitstream twice, even for streams created from constant floats in the original function.
We can also call `doit` repeatedly to continue to sample to computation after it has been set up.
This is very useful for doing monte carlo on the computation.
`doit` outputs a stochastic bitvector, so we use `from_stoch` to interpret it as a floating point.