import math
from numbers import Number
from typing import Union, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np

import numba

class PosStochasticComputing:
    APPLY_FLIPS = False
    APPLY_SHIFTS = False
    BITSHIFT_PROBABILITY = 0.01
    BITFLIP_PROBABILITY = 0.01

    @staticmethod
    def apply_bitshift(bitstream: np.ndarray):
        if not PosStochasticComputing.APPLY_SHIFTS:
            return bitstream

        # apply the bitshift error to the bitstream with probability 0.0001
        samples = np.random.uniform(0.0, 1.0, bitstream.shape)
        choice = samples < PosStochasticComputing.BITSHIFT_PROBABILITY
        locations = np.argwhere(choice)
        # print(f'Applied bitshift to {len(locations)} locations')
        return np.insert(bitstream, locations[:, 0], 0)[0:bitstream.shape[0]]


    @staticmethod
    def apply_bitflip(bitstream: np.ndarray):
        if not PosStochasticComputing.APPLY_FLIPS:
            return bitstream

        # apply the bitflip error to the bitstream with probability 0.0001
        samples = np.random.uniform(0.0, 1.0, bitstream.shape)
        choice = samples < PosStochasticComputing.BITFLIP_PROBABILITY
        # print(f'Applied bitflip to {np.sum(choice)} locations')
        return np.where(choice, np.logical_not(bitstream), bitstream)


    @staticmethod
    def to_stoch(prob: float, nbits: int) -> np.ndarray:
        """convert a decimal value in [0,1] to an <nbit> length bitstream."""
        assert 0.0 <= prob <= 1.0
        samples = np.random.uniform(0.0, 1.0, nbits)
        bitstream = samples < prob
        return bitstream


    @staticmethod
    def stoch_add(bitstream: np.ndarray, bitstream2: np.ndarray) -> np.ndarray:
        """add two stochastic bitstreams together"""
        assert (len(bitstream) == len(bitstream2))
        samples = np.random.uniform(0.0, 1.0, bitstream.shape)
        choice = samples > 0.5
        return np.where(choice, bitstream, bitstream2)

    @staticmethod
    def stoch_mul(bitstream, bitstream2):
        """multiply two stochastic bitstreams together"""
        assert (len(bitstream) == len(bitstream2))
        return bitstream & bitstream2

    @staticmethod
    def from_stoch(bitstream):
        """convert a stochastic bitstream to a numerical value"""
        return np.mean(bitstream, None, float)


@numba.experimental.jitclass([
    ('_precision_history', numba.types.List(numba.float64)),
])
class StochasticComputingStaticAnalysis:
    def __init__(self):
        self._precision_history = [float(0) for _ in range(0)]

    def req_length(self, smallest_value: float) -> int:
        """figure out the smallest bitstream length necessary represent the input decimal value. This is also called the precision."""
        return math.ceil(1.0 / smallest_value)

    def stoch_var(self, prec: float) -> float:
        """update static analysis -- the expression contains a variable with precision <prec>."""
        self._precision_history.append(prec)
        return prec

    def stoch_add(self, prec1: float, prec2: float) -> float:
        """update static analysis -- the expression adds together two bitstreams with precisions <prec1> and <prec2>."""
        result_prec = 1/2 * (prec1 + prec2)
        self.stoch_var(result_prec)
        return result_prec

    def stoch_mul(self, prec1: float, prec2: float) -> float:
        """update static analysis -- the expression multiplies together two bitstreams with precisions <prec1> and <prec2>."""
        result_prec = prec1 * prec2
        self.stoch_var(result_prec)
        return result_prec

    def get_size(self) -> int:
        """get minimum bitstream length required by computation."""
        # Explicit for numba compiler
        min_value = 9999999.0
        for prec in self._precision_history:
            if prec < min_value:
                min_value = prec
        return self.req_length(min_value)


# run a stochastic computation for ntrials trials
def run_stochastic_computation(lambd, ntrials, visualize=True, summary=True):
    results = []
    reference_value, _ = lambd()
    for i in range(ntrials):
        _, result = lambd()
        results.append(result)

    if visualize:
        nbins = math.floor(np.sqrt(ntrials))
        plt.hist(results, bins=nbins)
        plt.axvline(x=reference_value, color="red")
        plt.show()
    if summary:
        print("ref=%f" % (reference_value))
        print("mean=%f" % np.mean(results))
        print("std=%f" % np.std(results))


def PART_A_example_computation(bitstream_len):
    to_stoch = PosStochasticComputing.to_stoch
    stoch_mul = PosStochasticComputing.stoch_mul
    stoch_add = PosStochasticComputing.stoch_add
    apply_bitflip = PosStochasticComputing.apply_bitflip
    apply_bitshift = PosStochasticComputing.apply_bitshift

    # expression: 1/2*(0.8*0.4 + 0.6)
    reference_value = 1 / 2 * (0.8 * 0.4 + 0.6)

    w = apply_bitflip(apply_bitshift(to_stoch(0.8, bitstream_len)))
    x = apply_bitflip(apply_bitshift(to_stoch(0.4, bitstream_len)))
    y = apply_bitflip(apply_bitshift(to_stoch(0.6, bitstream_len)))
    tmp = apply_bitflip(apply_bitshift(stoch_mul(x, w)))
    result = apply_bitflip(apply_bitshift(stoch_add(tmp, y)))
    return reference_value, PosStochasticComputing.from_stoch(result)

def PART_A_low_precision(bitstream_len: int):
    to_stoch = PosStochasticComputing.to_stoch
    stoch_mul = PosStochasticComputing.stoch_mul
    # expression: 0.10 * 0.11 * 0.12 * 0.13
    reference_value = 0.353 * 0.5
    a = to_stoch(0.353, bitstream_len)
    b = to_stoch(0.5, bitstream_len)
    result = stoch_mul(a, b)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_analyze_wxb_function(precs):
    # w*x + b
    analysis = StochasticComputingStaticAnalysis()
    w_prec = analysis.stoch_var(precs["w"])
    x_prec = analysis.stoch_var(precs["x"])
    b_prec = analysis.stoch_var(precs["b"])
    res_prec = analysis.stoch_mul(w_prec, x_prec)
    analysis.stoch_add(res_prec, b_prec)
    N = analysis.get_size()
    print("best size: %d" % N)
    return N


def PART_Y_execute_wxb_function(values, N):
    # expression: 1/2*(0.8*0.4 + 0.6)
    w = values["w"]
    x = values["x"]
    b = values["b"]
    reference_value = 1 / 2 * (w * x + b)
    w = PosStochasticComputing.to_stoch(w, N)
    x = PosStochasticComputing.to_stoch(x, N)
    b = PosStochasticComputing.to_stoch(b, N)
    tmp = PosStochasticComputing.stoch_mul(x, w)
    result = PosStochasticComputing.stoch_add(tmp, b)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_test_analysis():
    precs = {"x": 0.1, "b": 0.1, "w": 0.01}
    # apply the static analysis to the w*x+b expression, where the precision of x and b is 0.1 and
    # the precision of w is 0.01
    analyze_example_computation = PART_Y_analyze_wxb_function
    execute_example_computation = PART_Y_execute_wxb_function

    N_optimal = analyze_example_computation(precs)
    print("best size: %d" % N_optimal)


    variables = {}
    for _ in range(10):
        variables["x"] = round(np.random.uniform(), 1)
        variables["w"] = round(np.random.uniform(), 2)
        variables["b"] = round(np.random.uniform(), 1)
        print(variables)
        run_stochastic_computation(lambda: execute_example_computation(variables, N_optimal), ntrials=10000,
                                   visualize=False)
        print("")


def PART_Z_execute_rng_efficient_computation(values, N, save_rngs=True):
    # expression: 1/2*(0.8*0.4 + 0.6)
    xv = values["x"]
    reference_value = 1 / 2 * (xv * xv + xv)
    if save_rngs:
        x = PosStochasticComputing.to_stoch(xv, N)
        x2 = x
        x3 = x
    else:
        x = PosStochasticComputing.to_stoch(xv, N)
        x2 = PosStochasticComputing.to_stoch(xv, N)
        x3 = PosStochasticComputing.to_stoch(xv, N)

    tmp = PosStochasticComputing.stoch_mul(x, x2)
    result = PosStochasticComputing.stoch_add(tmp, x3)
    return reference_value, PosStochasticComputing.from_stoch(result)

def PART_Z_execute_rng_efficient_computation_fixed(values, N, save_rngs=True):
    N = N + 2
    # expression: 1/2*(0.8*0.4 + 0.6)
    xv = values["x"]
    reference_value = 1 / 2 * (xv * xv + xv)
    if save_rngs:
        x = PosStochasticComputing.to_stoch(xv, N)
        x1 = x[0:-2]
        x2 = x[1:-1]
        x3 = x[2:]
    else:
        x1 = PosStochasticComputing.to_stoch(xv, N)
        x2 = PosStochasticComputing.to_stoch(xv, N)
        x3 = PosStochasticComputing.to_stoch(xv, N)

    tmp = PosStochasticComputing.stoch_mul(x1, x2)
    result = PosStochasticComputing.stoch_add(tmp, x3)
    return reference_value, PosStochasticComputing.from_stoch(result)


print("---- part a: effect of length on stochastic computation ---")
ntrials = 10000
# run_stochastic_computation(lambda: PART_A_example_computation(bitstream_len=10), ntrials)
# run_stochastic_computation(lambda: PART_A_example_computation(bitstream_len=100), ntrials)
run_stochastic_computation(lambda: PART_A_example_computation(bitstream_len=1000), ntrials)
#
# smallest_value = 1/1000
# run_stochastic_computation(lambda: (smallest_value, PosStochasticComputing.from_stoch(PosStochasticComputing.to_stoch(smallest_value, nbits=1000))), ntrials)
#
# run_stochastic_computation(lambda: PART_A_low_precision(bitstream_len=1000), ntrials)

# Part X, introduce non-idealities
PosStochasticComputing.APPLY_FLIPS = True
PosStochasticComputing.APPLY_SHIFTS = False
print("---- part x: effect of bit flips ---")
run_stochastic_computation(lambda: PART_A_example_computation(bitstream_len=1000), ntrials)
PosStochasticComputing.APPLY_FLIPS = False
PosStochasticComputing.APPLY_SHIFTS = True
print("---- part x: effect of bit shifts ---")
run_stochastic_computation(lambda: PART_A_example_computation(bitstream_len=1000), ntrials)
PosStochasticComputing.APPLY_FLIPS = False
PosStochasticComputing.APPLY_SHIFTS = False

# Part Y, apply static analysis
print("---- part y: apply static analysis ---")
PART_Y_test_analysis()

values = {"x": 0.00012, "b": 0.124, "w": 0.1}
print('Running bad example for Y')
execute_example_computation = lambda N: PART_Y_execute_wxb_function(values, N)
run_stochastic_computation(lambda: execute_example_computation(1000), ntrials=10000,
                           visualize=False)
print('Finished bad example for Y')

# Part Z, resource efficent rng generation
print("---- part z: one-rng optimization ---")
run_stochastic_computation(lambda: PART_Z_execute_rng_efficient_computation(values={'x': 0.3}, N=1000), ntrials)

print("---- part z: one-rng optimization, fixed ---")
run_stochastic_computation(lambda: PART_Z_execute_rng_efficient_computation_fixed(values={'x': 0.3}, N=1000), ntrials)


### Parsing and tracing Python ASTs for the static analysis
import ast
import inspect

class StaticAnalysisAST:
    class TracingPrecision:
        def __init__(self, min_value: float, parent: 'StaticAnalysisAST'):
            self.parent = parent
            self.min_value = min_value

        def __mul__(self, other: Union['StaticAnalysisAST.TracingPrecision', Number]) -> 'StaticAnalysisAST.TracingPrecision':
            if isinstance(other, StaticAnalysisAST.TracingPrecision):
                other = other.min_value
            return self.parent.create_value(self.min_value * other)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __add__(self, other: Union['StaticAnalysisAST.TracingPrecision', Number]) -> 'StaticAnalysisAST.TracingPrecision':
            if isinstance(other, StaticAnalysisAST.TracingPrecision):
                other = other.min_value
            return self.parent.create_value(1/2 * (self.min_value + other))

        def __radd__(self, other):
            return self.__add__(other)

    def __init__(self):
        self._precision_history = []

    def create_value(self, value: float) -> TracingPrecision:
        self._precision_history.append(value)
        return self.TracingPrecision(value, parent=self)

    def get_precision(self) -> float:
        return min(self._precision_history)

    def get_min_length(self) -> int:
        return math.ceil(1.0 / self.get_precision())

def ast_static_analysis(func: Callable, arg_precisions: Dict[str, float] = None):
    if arg_precisions is None:
        arg_precisions = {}
    # Use tracing to with new variable types to determine the precision required for each variable
    module_node = ast.parse(inspect.getsource(func))
    function_node = module_node.body[0]
    print(ast.dump(function_node))

    # Make sure we have all the arguments and no more
    assert set(arg.arg for arg in function_node.args.args) == set(arg_precisions.keys())

    sa = StaticAnalysisAST()
    precision_args = {arg: sa.create_value(prec) for arg, prec in arg_precisions.items()}

    # Compute the result to set the precisions throughout the data structure
    _result = python_function(**precision_args)

    # Don't care about the final result, just the minimum required precision

    return sa


class StochasticComputeTracing:
    NBITS = 10000
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __mul__(self, other):
        if not isinstance(other, StochasticComputeTracing):
            other = StochasticComputeTracing(op='constant', args=(other,))
        return StochasticComputeTracing(op='mul', args=(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if not isinstance(other, StochasticComputeTracing):
            other = StochasticComputeTracing(op='constant', args=(other,))
        return StochasticComputeTracing(op='add', args=(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def doit(self):
        if self.op == 'constant':
            return PosStochasticComputing.to_stoch(self.args[0], nbits=StochasticComputeTracing.NBITS)
        args_evaled = [arg.doit() for arg in self.args]
        if self.op == 'mul':
            return PosStochasticComputing.stoch_mul(*args_evaled)
        elif self.op == 'add':
            return PosStochasticComputing.stoch_add(*args_evaled)
        else:
            raise ValueError(f'Unknown op {self.op}')

    def __repr__(self):
        op = self.op
        args = self.args
        return f'StochasticComputeTracing({op=}, {args=})'


def build_ast_eval_computation(func: Callable, arg_values: Dict[str, float]) -> StochasticComputeTracing:
    tracing_arg_values = {arg: StochasticComputeTracing(op='constant', args=(value,)) for arg, value in arg_values.items()}
    result = func(**tracing_arg_values)
    return result


def python_function(x, y):
    # Should give 1/2 * (0.1 * 0.2 + 0.3 * 0.1) = 0.025
    return x * y + 0.3 * x


# Produce static analysis given the arbitrary python function
sa = ast_static_analysis(python_function, arg_precisions={'x': 0.1, 'y': 0.2})
# Print the evaluation metrics from our static analysis
print(f'{sa.get_precision() = }')
print(f'{sa.get_min_length() = }')

# Produce a stochastic computation given the arbitrary python function
eval_computation = build_ast_eval_computation(python_function, arg_values={'x': 0.1, 'y': 0.2})

# The AST for the stochastic computation can be inspected by printing the repr
print(repr(eval_computation))
# Finally we can run the computation to get an output value
print('Running traced computation:', PosStochasticComputing.from_stoch(eval_computation.doit()))
# Note that this will not reuse samples from the same random seed!

# The computation can also be repeated an arbitrary number of times
print('Running traced computation a second time:', PosStochasticComputing.from_stoch(eval_computation.doit()))