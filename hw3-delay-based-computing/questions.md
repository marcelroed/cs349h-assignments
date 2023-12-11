# Assignment: Delay-Based Computing

Implement the first arrival (FA), coincidence (COIN), and inhibition (INH) gates in `delay_gates.py`. To implement each gate, you need to fill in the `reset` and the `execute` functions:

 - The `reset` function should reset the state of the gate before execution. Recall delay-based computing gates are stateful and "remember" pulse information, and pulses are stored during a loading stage then released during execution. The reset function should define and reset whatever flags are necessary to correctly implement the operation of the gate.
 - The `execute` function should determine if the gate produces a pulse at a given timestep. The execute function takes as input the current time step (in nanoseconds), and a dictionary of input-value pairs. In this implementation, each port may have two values: PULSE (1) or NO_PULSE (0).

I have supplied the `delay_circuit.py` file, which simulates a delay-based computing circuit and renders the results, and a `delay_tester.py` python file which constructs some simple single-gate circuits and simulates them for you. Each test function in the `delay_tester.py` produces a visualization of each simulation and a visualization of the circuit architecture. The measured time window is shaded :

## Part X: Understanding Delay-Based Encodings [8 points]

Execute the `delay_tester.py` file as is to investigate the behavior of delay-based in input over time. The `test_input.py` function simulates the propagation of three constant delay-based values with labels ('X', 'Y', 'Z') through an empty circuit and outputs a pulse trace (`test_input.png`) and a summary of all the pulse arrivals/departures to console.

1. What does each numbered segment in the input image represent? What does it mean when a pulse (vertical line) is contained within a numbered segment?

2. What kind of value does an earlier pulse encode, what kind of value does a later pulse encode?

3. Run the simulation multiple times. Do you observe any variations in the plots from execution to execution?

4. Look at the pulse summary printed to console. How is the numerical value computed from the relative delay and segment time? How is the relative delay computed from the absolute delay and the gate's settling time? The settling time for a given port is the time required for the pulse to reach a particular port. 

# Part Y: Implementing the Delay Gates [8 pts]

Next, we will implement the first arrival, last arrival, delay, and inhibition gates associated with delay-based computing. Each delay gate is reset before execution, and then "executed" at each time step. The reset and execute functions may instantiate and change the gate's internal state, feel free to instantiate any internal variables needed.

1. Implement the reset and execute functions for the first arrival (`FirstArrival`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_first_arrival_gate` function to test the operation of the first arrival gate. 


2. Implement the reset and execute functions for the first arrival (`LastArrival`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_last_arrival_gate` function to test the operation of the last arrival gate. 


3. Implement the reset and execute functions for the first arrival (`Delay`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_delay_gate` function to test the operation of the delay gate. What does the circuit described in the `test_delay_gate` function do to the input value? 


4. Implement the reset and execute functions for the first arrival (`Inhibition`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_inh_gate` function to test the operation of the delay gate. 

# Part Z: Using Delay-Based Computing [6 pts]

Implement a circuit that computes the numerical function.  

    if max(X,Y) > 5:
        return max(X,Y) 
    else: 
        <no pulse>

Run it on a few values to test the function. You can use the `add_wire(src_gate,dst_gate,port)` function to add wires between gates, and the `add_gate(gate)` function to add a gate to the circuit. Feel free to use the `render_circuit("filename.png")` function to visualize the circuit that you built:

1. What is the structure of the circuit you implemented? [4 pts]

Let's say you want to use delay-based computation to implement operations in the log-domain. With this method, instead of computing directly over numeric values, you encode the log (e.g., `log2`(4) instead of 4) of the numeric value as a delay, and then use delay-based computing do implement the computation, and then you raise the delay-based value to `2` to recover the original value. For example, to implement X*Y, we do the following steps:

- convert to log-domain: lX = log2(X), lY = log2(Y)
- perform computation on log-domain values: lRes=lX+lY
- convert to value-domain: 2^{lRes}

2. What operations can you implement in the log-domain with delay-based computing? Are these operations implementable with delay-based computing if you were to directly encode the values as delays?

3. What operations are not implementable in the log-domain with delay-based computing? Are these computations implementable if we were to directly encode the variable values as delays?


# Part W: Digital Logic with Delay-Based Computing [14 pts]

Move to the `delay_digital.py` file. The following file uses race-based computing to execute `(X | not Y) and (Z | X) and (Z | Y)`. I have already implemented the scaffold for this part of the project which builds the logic circuit, and convenience functions for performing the dual-rail encoding.

1. Implement the `and` gate using delay logic. How did you implement this gate?

2. Implement the `or` gate using delay logic. How did you implement this gate?

3. Implement the `not` gate using delay logic. How did you implement this gate?

4. Implement the digital readout gate in `delay_gates.py` (see class `DigitalReadOutGate`). This gate should remember if it saw a pulse or not.

5. Execute the completed implementation on a few digital values. What delay-based encoding does a digital "1" output correspond to? What delay-based encoding does the digital "0" output correspond to? What states are invalid in the current digital value encoding?
 
6. Currently, the implementation creates a fresh input signal for every usage of every variable instead of re-using the input signal. Why is this necessary? What happens when you build the same circuit, but then reuse the same input for multiple gates.

7. What happens when you don't reset the circuit between executions? Why is a circuit reset operation necessary for delay-based digital logic, but not necessary for conventional digital logic?



