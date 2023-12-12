from tool.state_space.ss_sg import generate_linearization_point_SG, generate_SG_pu
from tool.state_space.ss_vsc import generate_linearization_point_VSC, generate_VSC_pu

def generate_SS_elements(d_grid, delta_slk, l_blocks, l_states):
    
    # SG
    
    lp_SG = generate_linearization_point_SG(d_grid)      
    l_blocks, l_states = generate_SG_pu(l_blocks, l_states, d_grid, lp_SG)
    
    # VSC
    
    lp_VSC = generate_linearization_point_VSC(d_grid)
    l_blocks, l_states = generate_VSC_pu(l_blocks, l_states, d_grid, lp_VSC)
       
    
    return l_blocks, l_states





