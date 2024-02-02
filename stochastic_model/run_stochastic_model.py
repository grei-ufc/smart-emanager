from generate_scenarios import main as update_scenarios
from networkx.convert_matrix import _generate_weighted_edges
import pyomo.pysp.util.rapper as rapper
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
import pyomo.environ as pyo

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import pathlib
import time
import sys
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

def main(data):
    retry = False
    while True:
        if retry:
            data['node_id'] += 1

        update_scenarios(scenarios_qtd=9, reduced_scenarios_qtd=3, data=data)
        solvername = 'cplex'
        path = str(pathlib.Path(__file__).parent.absolute())

        abstract_tree = CreateAbstractScenarioTreeModel()
        concrete_tree = abstract_tree.create_instance(
            path + '/ScenarioStructure.dat')

        stsolver = rapper.StochSolver(fsfile=path + '/ReferenceModel.py',
                                      # fsfct='pysp_instance_creation_callback',
                                      tree_model=concrete_tree)

        # ef_sol = stsolver.solve_ef(solvername)

        ef_sol = stsolver.solve_ef(solvername,
                                   generate_weighted_cvar=True,
                                   cvar_weight=1.0,
                                   risk_alpha=0.1)

        if ef_sol.solver.termination_condition != pyo.TerminationCondition.optimal:
            print('Not optimal solution: {}'.format(
                ef_sol.solver.termination_condition))
            retry = True
            continue
        else:
            retry = False
            print('Optimal solution finded: {}'.format(
                ef_sol.solver.termination_condition))
            # for varname, varvalue in stsolver.root_Var_solution():
            #     print(varname, str(varvalue))

        scenarios = stsolver.scenario_tree.scenarios

        model = rnd.choice(scenarios).instance
        break

    del scenarios
    del stsolver
    del ef_sol
    del abstract_tree
    del concrete_tree

    return model


if __name__ == '__main__':

    config = dict()
    config['has_storage'] = True
    config['storage_rate'] = 0.2
    config['storage_size'] = 10.0
    config['max_energy_flow'] = config['storage_rate'] * config['storage_size']
    config['max_soc'] = config['storage_size']
    config['min_soc'] = 0.1 * config['storage_size']

    config['bilateral_price'] = 38.0
    config['bilateral_max'] = 5.0

    config['node_id'] = 32
    config['load_kw'] = 4.0
    config['gen_kw'] = 1.0

    config['indice'] = (3488, '01/05/2019')
    
    config['price'] = [0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 
                       0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 
                       0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 
                       0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 
                       0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 
                       0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 1.004, 1.004, 1.004, 1.004, 1.582, 
                       1.582, 1.582, 1.582, 1.582, 1.582, 1.582, 1.582, 1.582, 1.582, 1.582, 1.582,1.004, 
                       1.004,1.004,1.004, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604, 0.604]

    config['prices'] = 0.703

    for i in range(1):
        config['node_id'] += 1
        model = main(data=config)
    
    soc_data = list(model.soc.get_values().values())
    p_spot_data = list(model.p_spot.get_values().values())
    spot_prices_data = [i.value for i in model.spot_prices.values()]
    p_bilateral_data = list(model.p_bilateral.get_values().values())
    demand_data = np.array([i.value for i in model.demand.values()])
    p_charge_data = np.array(list(model.p_charge.get_values().values()))
    p_discharge_data = np.array(list(model.p_discharge.get_values().values()))
    p_storage_data = p_charge_data - p_discharge_data
    demand_response_data = np.array([i.value for i in model.demand_response.values()])
    load_data = np.array([i.value for i in model.total_load.values()])
    #prices_data = np.array([model.prices[i]/100 for i in model.prices])

    time_data = [t / 4 for t in range(len(p_spot_data))]

    plt.plot(time_data, soc_data)
    plt.xlabel('Horas')
    plt.ylabel('Armazenamento (Ah)')
    plt.grid(True)
    plt.show()
    
    plt.plot(time_data, demand_data,
             time_data, p_storage_data,
             time_data, demand_data+p_storage_data-demand_response_data)
    plt.legend(['load+gen.', 'storage', 'load+gen+stor.'])
    plt.xlabel('Horas')
    plt.ylabel('Energia')
    plt.grid(True)
    plt.show()

    plt.plot(time_data, p_spot_data, time_data, p_bilateral_data)
    plt.legend(['spot', 'bilateral'])
    plt.xlabel('Horas')
    plt.ylabel('Energia')
    plt.grid(True)
    plt.show()
    

    plt.plot(time_data, demand_response_data,
             time_data, load_data)
    plt.legend(['demand_response', 'load'])
    plt.xlabel('Horas')
    plt.ylabel('Potência (kW)')
    plt.grid(True)
    plt.show()

    time_data = [t / 4 for t in range(len(spot_prices_data))]
    plt.plot(time_data, spot_prices_data)
    plt.xlabel('Horas')
    plt.ylabel('Preço (kWh)')
    plt.grid(True)
    plt.show()
    
    print(f'{sum(p_spot_data)} \n {sum(p_bilateral_data)} \n {sum(load_data)} \n {sum(demand_response_data)} \n {sum(spot_prices_data)} \n')
    
    
