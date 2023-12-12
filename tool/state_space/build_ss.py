import control as ct

def select_io(l_blocks,varin,varout):
    
    inputNames = []
    outputNames = []
    
    for ss_block in l_blocks:
        inputNames.extend(ss_block.input_labels)
        outputNames.extend(ss_block.output_labels)
    
    outputNames = sorted(list(set(outputNames)))    
    inputNames  = list(set(inputNames) - set(outputNames))
    inputNames = sorted(list(set(inputNames)))
    
    # Check if varin are external inputs
    valid_input = [var in inputNames for var in varin]
    if not all(valid_input):
        print("input is not valid")
        
    # Check if varout are external outputs
    if varout[0] == 'all':
        varout = outputNames
    else:
        valid_output = [var in outputNames for var in varout]
        if not all(valid_output):
            print("output is not valid")  

    return varin, varout 


def connect(l_blocks, l_states, inputs, outputs):    
    ss_sys = ct.interconnect(l_blocks, states = l_states, inputs = inputs, outputs = outputs, check_unused = False)    
    #ss_sys._remove_useless_states() 
    return ss_sys




