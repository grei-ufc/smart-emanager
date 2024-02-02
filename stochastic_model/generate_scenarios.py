import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pathlib
import sys

path = str(pathlib.Path().absolute())
sys.path.append(path)

from load_data import get_load_data, get_gen_data, get_spot_prices_data
from load_profile import demand_response_load, random_index
from forecast.price_fcst_sim import simulate_elect_price_fcst


def pick_load_scenarios_from_db(n, data=None):
    loads = list()

    if data:
        node_id = data['node_id']
        for scenario in range(n):
            load_df = get_load_data(load_index=node_id)[:96]
            normalized_df = (load_df-load_df.min())/(load_df.max()-load_df.min())
            load_df = data['load_kw'] * normalized_df
            loads.append(load_df.pload.values)
            node_id += 1
    else:
        for scenario in range(n):
            some_load_max_value = 2.5
            load_df = get_load_data()[:96]
            normalized_df = (load_df-load_df.min())/(load_df.max()-load_df.min())
            load_df = some_load_max_value * normalized_df
            loads.append(load_df.pload.values)
    return loads


def pick_gen_scenarios_from_db(n, data=None):
    gens = list()

    if data:
        node_id = data['node_id']
        for scenario in range(n):
            gen_df = get_gen_data(gen_index=node_id)[:96]
            normalized_df = (gen_df-gen_df.min())/(gen_df.max()-gen_df.min())
            gen_df = data['gen_kw'] * normalized_df
            gens.append(gen_df.pgen.values)
            node_id += 1
    else:
        for scenario in range(n):
            some_gen_max_value = 1.5
            gen_df = get_gen_data()[:96]
            normalized_df = (gen_df-gen_df.min())/(gen_df.max()-gen_df.min())
            gen_df = some_gen_max_value * normalized_df
            gens.append(gen_df.pgen.values)
    return gens


def pick_spot_prices_scenarios_from_db(n, data=None):
    prices = list()

    if data:
        node_id = data['node_id']
        for scenario in range(n):
            prices.append(get_spot_prices_data(spot_index=node_id).Price.values[:96])
            node_id += 1
    else:
        for scenario in range(n):
            prices.append(get_spot_prices_data().Price.values[:96])
    return prices


def pick_demand_response_scenarios_from_db(n, data=None):
    load = []
    if data:
        indices = data['indice']
        for scenarios in range(n):
            load.append(demand_response_load(indices))
            indices = random_index(indices)
    else:
        indices = random_index()
        for scenarios in range(n):
            load.append(demand_response_load(indices))
            indices = random_index(indices)

    return load


def demand_response_trat(scenarios):
    load = []
    for l in scenarios:
        aux = [0] * 96
        for j, k in l.items():
            if j != 'shift_duration':
                for m in k:
                    for n in range(len(aux)):
                        aux[n] = aux[n] + m[n]
        aux1 = np.array(aux)
        load.append(aux1)   

    return load


def reduce_scenarios(scenarios, reduced_scenarios_qtd, probs = None, demand=None):
    # ---------------------------------------------
    # Definição das variaveis utilizadas
    # ---------------------------------------------

    scenarios_qtd = len(scenarios)
    if not probs:
        probs = [1 / scenarios_qtd for i in range(scenarios_qtd)]
    
    reduced_load_scenarios = list()
    scenarios_reduced_index = list()
    kant_distances = list()
    costs = list()    
    
    for l in scenarios:
        aux = list()
        for k in scenarios:
            aux.append(np.linalg.norm(l - k))
        costs.append(aux)
    costs = np.array(costs)
        
    for line in costs:
        aux = 0.0
        for i, j in enumerate(line):
            aux += j * probs[i]
        kant_distances.append(aux)
    kant_distances = np.array(kant_distances)
    
    # Select o cenario com a minima distancia de Kantorovich 
    min_kant_dist_index = np.argmin(kant_distances)
    # Set e exclui o cenario para a lista de cenários reduzida
    reduced_load_scenarios.append(scenarios[min_kant_dist_index])
    # Set os custos do cenario encontrado 
    cost = costs[min_kant_dist_index, :]

    # Store o indice escolhido
    scenarios_reduced_index.append(min_kant_dist_index)
    
    # ---------------------------------------------
    # Compute a nova matriz de custos: step 2
    # ---------------------------------------------
    new_costs = costs
    new_kant_distances = kant_distances

    for _i in range(reduced_scenarios_qtd - 1):
        aux = list()
        for i, l in enumerate(costs):
            new_costs_line = list()
            if i not in scenarios_reduced_index:
                for k in l:
                    new_costs_line.append(min(cost[i], k))
            else:
                new_costs_line = new_costs[i, :]
            aux.append(new_costs_line)
        new_costs = np.array(aux)

        # Compute as novas distancias de Kantorovich
        aux1 = list()
        for scn_numb, line in enumerate(new_costs):
            aux2 = 0.0
            for i, j in enumerate(line):
                aux2 += j * probs[i]
            if scn_numb not in scenarios_reduced_index:
                aux1.append(aux2)
            else:
                aux1.append(np.inf)
        new_kant_distances = np.array(aux1)

        # Select o cenario com a minima distancia de Kantorovich 
        min_kant_dist_index = np.argmin(new_kant_distances)
        # Set e exclui o cenario para a lista de cenários reduzida
        reduced_load_scenarios.append(scenarios[min_kant_dist_index])
        # Set os custos do cenario encontrado 
        cost = new_costs[min_kant_dist_index, :]

        # Store o indice escolhido
        scenarios_reduced_index.append(min_kant_dist_index)

    # ---------------------------------------------
    # Compute as novas probabilidades dos cenarios: 
    # selecionados: Step 3
    # ---------------------------------------------
    probs_ = {i: probs[i] for i in scenarios_reduced_index}
    # loop entre os cenarios selecionados:
    for i in scenarios_reduced_index:
        # loop entre todos os cenarios:
        for j in range(scenarios_qtd):
            # se o cenario não é um dos cenarios escolhidos, então:
            if j not in scenarios_reduced_index:
                # loop entre os cenarios mais proximos de algum outro cenario, exceto ele mesmo
                for k in np.argsort(costs[j,:])[1:]:
                    # se o cenario mais proximo é o cenário atual, então atualiza a sua probabilidade
                    if k in scenarios_reduced_index: 
                        if k == i:
                            probs_[i] += probs[j]
                        break

    return probs_


def merge_scenarios(scenarios_data):
    merged_scenarios = list()
    dt1 = scenarios_data.pop(0)
    while scenarios_data:
        dt2 = scenarios_data.pop(0)
        for i1, i2 in dt1['probs'].items():
            for j1, j2 in dt2['probs'].items():
                new_scenario = {'name': 'Scenario-{}{}_{}{}'.format(dt1['prefix'], i1, dt2['prefix'], j1),
                                'prob': i2 * j2,
                                'data': [dt1['data'][i1], dt2['data'][j1]]}
                merged_scenarios.append(new_scenario)
    return merged_scenarios


def write_scenario_struct(data):
    has_storage = data['has_storage']
    data = data['scenarios']

    path = str(pathlib.Path(__file__).parent.absolute())

    with open(path + '/ScenarioStructure.dat', 'w') as f:
        f.write('set Stages := FirstStage SecondStage ;\n\n')

        f.write('set Nodes := RootNode\n')
        f.writelines(['{}-Node\n'.format(i) for i in data.keys()])
        f.write(' ;\n\n')

        f.write('param NodeStage := RootNode FirstStage\n')
        f.writelines(['{}-Node SecondStage\n'.format(i) for i in data.keys()])
        f.write(' ;\n\n')

        f.write('set Children[RootNode] := ')
        f.writelines(['{}-Node\n'.format(i) for i in data.keys()])
        f.write(' ;\n\n')

        f.write('param ConditionalProbability := RootNode 1.0\n')
        f.writelines(
            ['{}-Node {}\n'.format(i, round(data[i]['prob'], 8))
             for i in data.keys()])
        f.write(' ;\n\n')

        f.write('set Scenarios :=\n')
        f.writelines(['{}\n'.format(i) for i in data.keys()])
        f.write(' ;\n\n')

        f.write('param ScenarioLeafNode :=\n')
        f.writelines(['{} {}-Node\n'.format(i, i) for i in data.keys()])
        f.write(' ;\n\n')

        if has_storage:
            f.write('set StageVariables[FirstStage] := p_bilateral\nsoc[*] ;\n\n')
        else:
            f.write('set StageVariables[FirstStage] := p_bilateral\nsoc[*] ;\n\n')
            
        f.write('set StageVariables[SecondStage] := p_spot[*] ;\n\n')


        f.write('param StageCost := FirstStage  StageCost\nSecondStage StageCost ;')


def main(scenarios_qtd, reduced_scenarios_qtd, data=None):
    scenarios_qtd = scenarios_qtd  # numero de cenarios carregados
    reduced_scenarios_qtd = reduced_scenarios_qtd  # numero de cenarios reduzidos
    # ---------------------------------------------
    # carrega os dados dos cenarios de carga,
    # geração e preços do mercado spot
    # ---------------------------------------------
    if data:
        load_scenarios = pick_load_scenarios_from_db(scenarios_qtd, data)
        gen_scenarios = pick_gen_scenarios_from_db(scenarios_qtd, data)
        spot_prices_scenarios = pick_spot_prices_scenarios_from_db(scenarios_qtd, data)
        demand_response_scenarios = pick_demand_response_scenarios_from_db(scenarios_qtd, data)
    else:
        load_scenarios = pick_load_scenarios_from_db(scenarios_qtd)
        gen_scenarios = pick_gen_scenarios_from_db(scenarios_qtd)
        spot_prices_scenarios = pick_spot_prices_scenarios_from_db(scenarios_qtd)
        demand_response_scenarios = pick_demand_response_scenarios_from_db(scenarios_qtd)
        
    # ---------------------------------------------
    # tratamento dos dados para resposta da demanda
    # ---------------------------------------------
    
    demand_response = demand_response_trat(demand_response_scenarios)

    # ---------------------------------------------
    # reduz os cenarios de carga, geração e preços
    # ---------------------------------------------
    
    load_probs = reduce_scenarios(load_scenarios, reduced_scenarios_qtd)
    gen_probs = reduce_scenarios(gen_scenarios, reduced_scenarios_qtd)
    spot_prices_probs = reduce_scenarios(spot_prices_scenarios, reduced_scenarios_qtd)
    demand_response_probs = reduce_scenarios(demand_response, reduced_scenarios_qtd)

    # ---------------------------------------------
    # combina os cenarios de carga e de geração (demanda)
    # ---------------------------------------------
    scenarios_data = [{
        'prefix': 'load',
        'data': load_scenarios,
        'probs': load_probs,
    },
        {
            'prefix': 'gen',
            'data': gen_scenarios,
            'probs': gen_probs,
        }]
    merged_scenarios_data = merge_scenarios(list(scenarios_data))

    demand_scenarios = list()
    demand_probs = list()
    for i in merged_scenarios_data:
        i['data'] = i['data'][0] - i['data'][1]
        demand_scenarios.append(i['data'])
        demand_probs.append(i['prob'])
    demand_probs = {i: j for i, j in enumerate(demand_probs)}

    # ---------------------------------------------
    # reduz os cenarios de carga combinados com geração (demanda)
    # ---------------------------------------------
    reduced_demand_probs = reduce_scenarios(demand_scenarios,
                                            reduced_scenarios_qtd,
                                            demand_probs)

    # ---------------------------------------------
    # combina os cenarios de demanda e de preços spot
    # ---------------------------------------------
    scenarios_data = [{
        'prefix': 'demand',
        'data': demand_scenarios,
        'probs': reduced_demand_probs,
    },
        {
            'prefix': 'spot',
            'data': spot_prices_scenarios,
            'probs': spot_prices_probs,
        }]

    merged_scenarios_data = merge_scenarios(list(scenarios_data))

    data_demand_spot = []
    for k in merged_scenarios_data:
        data_demand_spot.append((k['data'][0], k['data'][1]))

    aux = [0]*96
    demand_spot_scenarios = []
    demand_spot_probs = list()
    for i in merged_scenarios_data:
        i['data'] = i['data'][0] - i['data'][1]
        demand_spot_scenarios.append(aux)
        demand_spot_probs.append(i['prob'])
    demand_spot_scenarios = np.array(demand_spot_scenarios)
    demand_spot_probs = {i: j for i, j in enumerate(demand_spot_probs)}
 
    reduced_demand_probs = reduce_scenarios(demand_spot_scenarios,
                                            reduced_scenarios_qtd,
                                            demand_spot_probs)
    
    scenarios_data = [{
        'data': data_demand_spot,
        'probs': reduced_demand_probs,
    },
        {
            'data': demand_response_scenarios,
            'probs': demand_response_probs,
        }]

    scenarios = list()
    dt1 = scenarios_data.pop(0)
    while scenarios_data:
        dt2 = scenarios_data.pop(0)
        for i1, i2 in dt1['probs'].items():
            name = merged_scenarios_data[i1]
            name = name['name']
            for j1, j2 in dt2['probs'].items():
                new_scenario = {'name': f'{name}_demand_response{j1}',
                                'prob': i2 * j2,
                                'data': [dt1['data'][i1], dt2['data'][j1]]}
                scenarios.append(new_scenario)
                
    prices = simulate_elect_price_fcst(rtp_input_data_path='D:\Desktop\Codigo\\forecast\input\RTP\\',
                                       t_start=pd.Timestamp('2017-4-30 00:00'),
                                       t_end=pd.Timestamp('2017-4-30 23:45'),
                                       pr_constant=0.25, pricing={'ToU', 'Random', 'ToU_mi', 'RTP'})

    if not data:
        data = dict()
        data['bilateral_price'] = 40.0
        data['bilateral_max'] = 2.0
        data['has_storage'] = True
        data['storage_rate'] = 0.1
        data['storage_size'] = 2.0
        data['max_energy_flow'] = 0.2
        data['max_soc'] = 2.0
        data['min_soc'] = 0.2
    
    data['prices_curt'] = [20, 25, 45, 35, 30, 50, 20, 25, 45, 35, 30, 50]
    data['prices_inte'] = [40, 35, 65, 70, 65, 80, 40, 35, 65, 70, 65, 80]
    data['prices_shift'] = [10, 20, 15, 25, 30, 20, 10, 20, 15, 25, 30, 20]
    
    data['prices'] = prices['ToU'].tolist()
    
    data['scenarios'] = {}
    for i in scenarios:
        aux = i['data'][1]
        data['scenarios'][i['name']] = {
            'prob': i['prob'],
            'demand_data': list(np.round(i['data'][0][0], 4)),
            'spot_price_data': list(np.round(i['data'][0][1], 4)),
        }
        for k in aux:
            if k == 'curt_load':
                data['scenarios'][i['name']].update({'curtailable_load': aux['curt_load']})
            if k == 'inte_load':
                data['scenarios'][i['name']].update({'interruptible_load': aux['inte_load']})
            if k == 'shift_load':
                data['scenarios'][i['name']].update({'shiftable_load': aux['shift_load']})
            if k == 'shift_duration':
                data['scenarios'][i['name']].update({'shiftable_duration': aux['shift_duration']})


    path = str(pathlib.Path(__file__).parent.absolute())
    with open(path + '/config.json', 'w') as f:
        json.dump(data, f)

    write_scenario_struct(data)

    del data
    del load_scenarios
    del gen_scenarios
    del spot_prices_scenarios
    del demand_scenarios

    del load_probs
    del gen_probs
    del spot_prices_probs
    del demand_probs

    del merged_scenarios_data


if __name__ == "__main__":
    main(scenarios_qtd=10, reduced_scenarios_qtd=3)
