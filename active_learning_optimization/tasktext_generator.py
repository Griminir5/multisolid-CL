import numpy as np
import numpy.random as rand
import random


num_tasks = 5


def generate_tasktext(max_steps, max_steptime):
    total_steps = rand.randint(4, max_steps)
    start_time = rand.randint(200,1001)
    covertext = 'Task CycleOrg runs when time > {}'.format(start_time)
    steps = []
    steps.append(covertext)
    compositions=np.array([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9])
    #print(compositions)
    temp=0
    flow=0

    for _ in range(total_steps):
        step_time = rand.randint(50, max_steptime)
        

        change_comp = bool(random.getrandbits(1)) or bool(random.getrandbits(1))
        change_temp = bool(random.getrandbits(1)) and bool(random.getrandbits(1))
        change_flow = bool(random.getrandbits(1))
        if not(change_comp) and not(change_temp) and not(change_flow):
            steps.append( 'wait {:.0f};'.format(step_time) )
        else:
            steps.append('PARALLEL')

            if change_comp: # whether composition is changed across this step
                have_oxygen = rand.uniform(low=0, high=1)
                print('Oxygen control - ', have_oxygen)
                if have_oxygen<=0.01:
                    compositions[[0,6]] = 0
                    compositions[[1,2,3,4,5,7,8]] = rand.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                    while (compositions[8]>0.21 or compositions[8]<0.05):
                        compositions[[1,2,3,4,5,7,8]] = rand.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                else: 
                    if have_oxygen<0.31:
                        compositions[[0,1,2,4,6]] = 0
                        compositions[[3,5,7,8]] = rand.dirichlet([0.5, 0.5, 0.5, 0.5])
                        
                        while (compositions[8]>0.21 or compositions[8]<0.15):
                            compositions[[3,5,7,8]] = rand.dirichlet([0.5, 0.5, 0.5, 0.5])
                    else:
                        compositions[[0,6,7,8]] = 0
                        compositions[[1,2,3,4,5]] = rand.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5])



                steps.append( 'SRamp(B1.gas_in.y("AR"),   {:.3f}, {:.0f});'.format(compositions[0], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("CH4"),  {:.3f}, {:.0f});'.format(compositions[1], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("CO"),   {:.3f}, {:.0f});'.format(compositions[2], step_time) ) 
                steps.append( 'SRamp(B1.gas_in.y("CO2"),  {:.3f}, {:.0f});'.format(compositions[3], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("H2"),   {:.3f}, {:.0f});'.format(compositions[4], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("H2O"),  {:.3f}, {:.0f});'.format(compositions[5], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("HE"),   {:.3f}, {:.0f});'.format(compositions[6], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("N2"),   {:.3f}, {:.0f});'.format(compositions[7], step_time) )
                steps.append( 'SRamp(B1.gas_in.y("O2"),   {:.3f}, {:.0f});'.format(compositions[8], step_time) )
            else :
                steps.append( '// SRamp(B1.gas_in.y("AR"),   {:.3f}, {:.0f});'.format(compositions[0], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("CH4"),  {:.3f}, {:.0f});'.format(compositions[1], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("CO"),   {:.3f}, {:.0f});'.format(compositions[2], step_time) ) 
                steps.append( '// SRamp(B1.gas_in.y("CO2"),  {:.3f}, {:.0f});'.format(compositions[3], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("H2"),   {:.3f}, {:.0f});'.format(compositions[4], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("H2O"),  {:.3f}, {:.0f});'.format(compositions[5], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("HE"),   {:.3f}, {:.0f});'.format(compositions[6], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("N2"),   {:.3f}, {:.0f});'.format(compositions[7], step_time) )
                steps.append( '// SRamp(B1.gas_in.y("O2"),   {:.3f}, {:.0f});'.format(compositions[8], step_time) )   


            if change_temp: # whether temperature is changed across the steps
                temp = rand.uniform(low=300, high=700)

                steps.append( 'SRamp(B1.gas_in.T, {:.0f}, {:.0f});'.format(temp, step_time) )
            else:
                steps.append( '// SRamp(B1.gas_in.T, {:.0f}, {:.0f});'.format(temp, step_time) )


            if change_flow: # whether flowrate is changed across the steps
                flow = rand.uniform(low=0.5, high=2)

                steps.append( 'SRamp(B1.gas_in.F, {:.3f}*B1.Fsteady, {:.0f});'.format(flow, step_time) )
            else:
                steps.append( '// SRamp(B1.gas_in.F, {:.3f}*B1.Fsteady, {:.0f});'.format(flow, step_time) )
            
            steps.append('ENDPARALLEL')
    steps.append('RESTART;')
    steps.append('End')

    return steps

for i in range(num_tasks):
    text = generate_tasktext(max_steps=100, max_steptime=500)
    #print(text)
    with open('tasknum{:.0f}.txt'.format(9900+i+5), 'w') as f:
        for line in text:
            f.write(f'{line}\n')

