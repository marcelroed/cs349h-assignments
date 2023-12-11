import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import collections as matcoll
import numpy as np
import graphviz
from delay_gates import *



class DelayBasedCircuit:

    def __init__(self, pulse_only=False):
        self.gates = {}
        self.wires = {}
        self.rev_wires = {}
        self.inputs = {}

        self.window_size=1e-9
        self.segments=10
        self.timestep=1e-12
        self.segment_time = self.window_size/(self.segments+1)
        self.pulse_only = pulse_only

    def add_gate(self,g):
        self.gates[g.id] = g
        if isinstance(g, Input):
            self.inputs[g.name] = g

        # source to sink wires
        self.rev_wires[g.id] = []
        return g
    
    def add_wire(self,g1,g2,port):
        assert(isinstance(g1,Gate))
        assert(isinstance(g2,Gate))
        self.wires[g1.id] = (g2.id,port)
        self.rev_wires[g2.id].append((g1.id,port))

    def get_sinks(self,g):
        return [] if not g.id in self.wires else [self.wires[g.id]]

    def get_inputs(self):
        return filter(lambda g: isinstance(g, Input), self.gates.values())

    def get_input(self,name):
        return self.inputs[name]

    def compute_settling_times(self):
        def _recurse(times,gate):
            # if there are no parents, this gate is toplevel, just return delay
            if len(self.rev_wires[gate.id]) == 0:
                times[(gate.id,OUT_NAME)] = gate.delay()
                return gate.delay()

            # get delays of upstream gates
            dels = []
            for src_gate,dst_port in self.rev_wires[gate.id]:
                dels.append(_recurse(times,self.gates[src_gate]))
                times[(gate.id,dst_port)] = max(dels)

            # the path must be balanced
            if not (all(x==dels[0] for x in dels)) and not self.pulse_only:
                raise Exception("gate %s: input circuit paths have uneven delays: %s" % (gate.id, dels))
            
            # compute maximum delay of this gate.
            times[(gate.id,OUT_NAME)] = gate.delay()+max(dels)
            return times[(gate.id,OUT_NAME)]

        self.settling_times = {}
        for g in self.gates.values():
            if g.id in self.wires:
                continue
            _recurse(self.settling_times,g)

    def gate_settling_time(self,gate,port):
        return self.settling_times[(gate.id,port)]

    def max_settling_time(self):
        return max(self.settling_times.values())

    def render_circuit(self,name):
        g = graphviz.Digraph()
        for gate in self.gates.values():
            input_terms = []
            for p in gate.ports:
                input_terms.append("<%s> %s" % (p,p))

            inp_str = "{%s}" % ("|".join(input_terms))
            if len(input_terms) > 0:
                label = "{inp_str} | <{gate_id}> {gate_id} | <out> out".format(inp_str=inp_str,gate_id=gate.id)
            else:
                label = "<{gate_id}> {gate_id} | <out> out".format(inp_str=inp_str,gate_id=gate.id)

            g.node(gate.id, label=label, shape="record")

        for gsrc,(gdst,port) in self.wires.items():
            g.edge(gsrc,gdst,tailport="out", headport=port)

        g.render(name, format="png")



    def render(self,filename,timing,traces):
        nplots = len(list(traces.keys()))

        fig, ax = plt.subplots(nplots,1,sharex=True)
        window_height = 1.2
        for idx,(name,data) in enumerate(traces.items()):
            gate,port = name
            xs = list(map(lambda d: d[0], data))
            ys = list(map(lambda d: d[1], data))
            settle = self.gate_settling_time(self.gates[gate],port)
            
            settle_rect = matplotlib.patches.Rectangle((0,0), 
                                        settle, window_height, 
                                        color ='grey', alpha=0.2) 
            measure_rect= matplotlib.patches.Rectangle((settle,0), 
                                    timing["measure"], window_height, 
                                    color ='green', alpha=0.2) 

            seglen = timing["segment"]
            meas_lines = list(map(lambda seg: [(settle+seg*seglen,0), (settle+seg*seglen,window_height)], \
                range(timing["n_segments"])))
            meas_linecoll = matcoll.LineCollection(meas_lines, colors='green',alpha=0.5)


            pulse_lines = list(map(lambda coord: [(coord[0],0),(coord[0],coord[1])], data))
            pulse_linecoll = matcoll.LineCollection(pulse_lines, colors='k')
            
            ax[idx].add_patch(measure_rect)
            ax[idx].add_collection(meas_linecoll)
            ax[idx].add_patch(settle_rect)
            for i in range(timing["n_segments"]):
                ax[idx].text(settle+(i+0.5)*seglen, 0.2, str(i), color="green",alpha=0.7)

            ax[idx].scatter(xs,ys)
            ax[idx].add_collection(pulse_linecoll)
            ax[idx].set_title("%s.%s" % (gate,port))

        plt.xlim(0, timing["max_time"])
        plt.ylim(0, 1.2)
        plt.xlabel("time (ns)")
        fig.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    def get_outputs(self,timing,traces):
        output_trace = []
        segment = timing["segment"]
        n_segments = timing["n_segments"]
        for idx,(name,data) in enumerate(traces.items()):
            gate,port = name
            xs = list(map(lambda d: d[0], data))
            ys = list(map(lambda d: d[1], data))
            settle = self.gate_settling_time(self.gates[gate],port)
            for time,_ in data:
                value = math.floor((time-settle)/segment)
                if value > n_segments:
                    value = "inf"
                if value < 0:
                    value = "invalid (neg)"
                
                output_trace.append((name, value, settle , segment, time, time-settle))

        return output_trace
        
    def simulate(self,inputs):
        traces = {}
        for g in self.gates.values():
            traces[(g.id,OUT_NAME)] = []
            for p in g.ports:
                traces[(g.id,p)] = []

        

        print("-> instantiating values") 
        for inp_name,value in inputs.items():
            if value is None:
                self.inputs[inp_name].set_no_pulse()

            else:
                tmin= self.segment_time*value
                self.inputs[inp_name].set_pulse_window(tmin,tmin+self.segment_time)

        self.compute_settling_times()
        timing = {}
        timing["measure"] = self.window_size
        timing["settle"] = self.max_settling_time() 
        timing["segment"] = self.segment_time 
        timing["n_segments"] = self.segments
        
        tmax = timing["settle"]+timing["measure"]
        timing["max_time"] = tmax 

        curr_tick = 0
        max_ticks = tmax/self.timestep


        #reset circuit
        print("-> resetting circuit") 
        for gate in self.gates.values():
            gate.reset()

        print("-> executing circuit") 
        pulse_buffer = {}
        pulse_buffer[0] = {}
        while curr_tick <= max_ticks:
            
            for key, gate in self.gates.items():
                # retrieve relevant inputs for gate
                gate_inputs = {}
                for port in gate.ports:
                    gate_inputs[port] = PULSE if (gate.id,port) \
                    in pulse_buffer[curr_tick] else NO_PULSE

                #add output pulses to buffer 
                if gate.execute(curr_tick*self.timestep, gate_inputs) == PULSE:
                    tick = int(math.ceil(gate.delay()/self.timestep))
                    traces[(gate.id,OUT_NAME)].append(((curr_tick+tick)*self.timestep, PULSE))

                    # add pulse to pulse buffer at appropriate tick
                    if not curr_tick + tick in pulse_buffer:
                        pulse_buffer[curr_tick + tick] = []

                    for sink_gate_id,sink_port in self.get_sinks(gate):
                        pulse_buffer[curr_tick + tick].append((sink_gate_id,sink_port))
                        traces[(sink_gate_id,sink_port)].append(((curr_tick+tick)*self.timestep, PULSE))

            del pulse_buffer[curr_tick]
            if not curr_tick + 1 in pulse_buffer:
                pulse_buffer[curr_tick + 1] = {}

            curr_tick += 1

        return timing,traces


