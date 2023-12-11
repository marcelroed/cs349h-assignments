


from delay_gates import *
from delay_circuit import *

def summarize_outputs(outputs):
    print("------")
    for (gate,port), value, settle, segment, abs_t, rel in outputs:
        print("%s port:%s = %s" % (gate,port,value))
        print("   settling time =\t%e" % (settle))
        print("   relative delay =\t%e" % (rel))
        print("   absolute delay =\t%e" % (abs_t))
        print("   segment time =\t%e" % (segment) )
        print("")

def test_input():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    inx = circ.add_gate(Input("Y"))
    inx = circ.add_gate(Input("Z"))
    timing, traces = circ.simulate({"X":4, "Y":7, "Z":2})
    circ.render("test_input.png",timing,traces)
    circ.render_circuit("test_input_circ.png")
    summarize_outputs(circ.get_outputs(timing,traces))


def test_last_arrival_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    la = circ.add_gate(LastArrival())
    circ.add_wire(inx,la,"A")
    circ.add_wire(iny,la,"B")

    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_la.png",timing,traces)
    circ.render_circuit("test_la_circ.png")
    summarize_outputs(circ.get_outputs(timing,traces))

def test_first_arrival_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    fa = circ.add_gate(FirstArrival())
    circ.add_wire(inx,fa,"A")
    circ.add_wire(iny,fa,"B")

    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_fa.png",timing,traces)
    circ.render_circuit("test_fa_circ.png")
    summarize_outputs(circ.get_outputs(timing,traces))


def test_delay_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    delay = circ.add_gate(DelayGate(2*circ.segment_time))
    circ.add_wire(inx,delay,"A")

    circ.render_circuit("test_del_circ.png")
    timing,traces = circ.simulate({"X":4})
    circ.render("test_del1.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    timing,traces = circ.simulate({"X":6})
    circ.render("test_del2.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))




def test_inh_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    inh = circ.add_gate(Inhibition())
    circ.add_wire(inx,inh,"A")
    circ.add_wire(iny,inh,"B")

    circ.render_circuit("test_inh_circ.png")
    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_inh1.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    timing,traces = circ.simulate({"X":7,"Y":4})
    circ.render("test_inh2.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))




test_input()
#test_first_arrival_gate()
#test_last_arrival_gate()
#test_inh_gate()
#test_delay_gate()
