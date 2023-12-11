from delay_gates import *
from delay_circuit import *

   

def bin_input(circ,name):
    pos_lbl = "%s_pos" % name
    neg_lbl = "%s_neg" % name
    pos_gate = circ.add_gate(Input(pos_lbl))
    neg_gate = circ.add_gate(Input(neg_lbl))
    return (pos_gate,neg_gate)

def bin_to_delay(name,val,segments=10):
    if val == 1: 
        pos = random.choice(list(range(segments)))
        neg = None
    else:
        pos = None
        neg = random.choice(list(range(segments)))
    
    pos_lbl = "%s_pos" % name
    neg_lbl = "%s_neg" % name

    yield (pos_lbl,pos)
    yield (neg_lbl,neg)

def and_gate(circ,A,B):
    (Ap,An) = A
    (Bp,Bn) = B
    Rp,Rn = None,None
    raise NotImplementedError 
    return (Rp,Rn)
    

def or_gate(circ,A,B):
    (Ap,An) = A
    (Bp,Bn) = B
    Rp,Rn = None,None
    raise NotImplementedError 
    return (Rp,Rn)
 


def not_gate(circ,A):
    (Ap,An) = A
    Rp,Rn = None,None
    raise NotImplementedError 
    return (Rp,Rn)


def build_logic_circuit():
    circ = DelayBasedCircuit(pulse_only=True)

    X1 = bin_input(circ,"X1")
    Y1 = bin_input(circ,"Y1")
    Z1 = bin_input(circ,"Z1")

    X2 = bin_input(circ,"X2")
    Y2 = bin_input(circ,"Y2")
    Z2 = bin_input(circ,"Z2")


    c1 = or_gate(circ,X1,not_gate(circ,Y1)) 
    c2 = or_gate(circ,Z1,X2) 
    c3 = or_gate(circ,Z2,Y2) 
    r = and_gate(circ,c1,c2)
    fr = and_gate(circ,r,c3)
    #fr_pos,fr_neg = fr
    
    dro_p = circ.add_gate(DigitalReadOutGate())
    dro_n = circ.add_gate(DigitalReadOutGate())
    circ.add_wire(fr[0],dro_p,"A")
    circ.add_wire(fr[1],dro_n,"A")

    return (dro_p,dro_n),circ


def execute_logic_circuit(dro,circ,Xbin,Ybin,Zbin):
    circ.render_circuit("digital_circuit.png")
   
    inps = []
    inps += bin_to_delay("X1",Xbin)
    inps += bin_to_delay("Y1",Ybin)
    inps += bin_to_delay("Z1",Zbin)
    
    inps += bin_to_delay("X2",Xbin)
    inps += bin_to_delay("Y2",Ybin)
    inps += bin_to_delay("Z2",Zbin)

    input_dict = dict(inps)

    timing,traces = circ.simulate(input_dict)
    circ.render("test_digital.png",timing,traces)
    dro_p, dro_n = dro
    if dro_p.has_pulse and not dro_n.has_pulse:
        result = 1
    elif not dro_p.has_pulse and dro_n.has_pulse:
        result = 0 
    else:
        raise Exception("invalid state: pos=%s, neg=%s" % (dro_p.has_pulse, dro_n.has_pulse))
    c1=(Xbin or (not Ybin))
    c2 = (Zbin or Xbin)
    c3 = (Zbin or Ybin)
    reference = c1 and c2 and c3

    print("computed: %d" % result)
    print("expected: %d" % (1 if reference else 0))

fr,circ = build_logic_circuit()
execute_logic_circuit(fr,circ,Xbin=1,Ybin=0,Zbin=1)

