# This is a program for testing the RSSNP simulator

from abstracts.grapher import draw
from abstracts.rssnp import assign_rssnp
from RSSNP_list import *

rssnp = assign_rssnp(add_rssnp_minimal)
draw(rssnp, "oop")

rssnp.in_spiketrain = [
    {
        'index': 0,
        'input': [0,0,1,0,0,0,1,1,1]
    },
    {
        'index': 1,
        'input': [1,0,1,1,0,0,0,1,1]
    },
    
]
print(rssnp.isValid())
print(rssnp.main((add_rssnp_minimal['init_config'],add_rssnp_minimal['rule_status'])))