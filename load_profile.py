"""
Este arquivo carrega perfis de carga curtailable, interruptible e shiftable
"""

import random as rn
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

path = os.path.dirname(os.path.abspath(__file__))

DEMAND_RESPONSE = path + '/15minute_data_newyork'

curtailable_load_df = pd.read_csv(f'{DEMAND_RESPONSE}/15minute_data_curtailable_newyork.csv',
                                  delimiter=';', parse_dates=['local_15min'],
                                  index_col=['dataid', 'local_15min'])

interruptible_load_df = pd.read_csv(f'{DEMAND_RESPONSE}/15minute_data_interruptible_newyork.csv',
                                    delimiter=';', parse_dates=['local_15min'],
                                    index_col=['dataid', 'local_15min'])

shiftable_load_df = pd.read_csv(f'{DEMAND_RESPONSE}/15minute_data_shiftable_newyork.csv',
                                delimiter=';', parse_dates=['local_15min'],
                                index_col=['dataid', 'local_15min'])


def random_index(index=None):
    """
        Gera o indice para a escolha dos dados
    """
    if index:
        while True:
            day = rn.randint(1, 30)
            mouth = rn.randint(5, 10)
            date = f'{day}/{mouth}/2019'

            list_house = [27,  142,  387,  558,  914,  950, 1222, 1240, 1417, 2096, 2318,
                          2358, 3000, 3488, 3517, 3700, 3996, 4283, 4550, 5058, 5587, 5679,
                          5982, 5997, 9053]

            house = rn.choice(list_house)

            indice = (house, date)

            if indice != index:
                break
    else:
        day = rn.randint(1, 30)
        mouth = rn.randint(5, 10)
        date = f'{day}/{mouth}/2019'

        list_house = [27,  142,  387,  558,  914,  950, 1222, 1240, 1417, 2096, 2318,
                      2358, 3000, 3488, 3517, 3700, 3996, 4283, 4550, 5058, 5587, 5679,
                      5982, 5997, 9053]

        house = rn.choice(list_house)

        indice = (house, date)

    return indice


def demand_response_load(index=None):
    """
        Recolhe os dados para a geração de cénarios
    """

    if index is None:
        index = random_index()

    curtailable_load = curtailable_load_df.loc[index]
    interruptable_load = interruptible_load_df.loc[index]
    shiftable_load = shiftable_load_df.loc[index]

    curtailable_load.dropna(axis=1, how='all', inplace=True)
    interruptable_load.dropna(axis=1, how='all', inplace=True)
    shiftable_load.dropna(axis=1, how='all', inplace=True)

    load = {'curt_load': [], 'inte_load': [], 'shift_load': [], 'shift_duration': []}

    load_aux = load.copy()

    for i in curtailable_load:
        aux = sum(curtailable_load[i].tolist())
        #if aux > 0:
        load['curt_load'].append(curtailable_load[i].tolist())

    for i in interruptable_load:
        aux = sum(interruptable_load[i].tolist())
        #if aux > 0:
        load['inte_load'].append(interruptable_load[i].tolist())

    for i in shiftable_load:
        aux = sum(shiftable_load[i].tolist())
        #if aux > 0:
        load['shift_load'].append(shiftable_load[i].tolist())
        aux_time = []
        for k in shiftable_load[i].tolist():
            if k != 0:
                aux_time.append(1)
            else:
                aux_time.append(0)
        load['shift_duration'].append(aux_time)

    """ aux_load = []
        
    for i in load_aux.keys():
        if not load[i]:
            aux_load.append(load.pop(i, None)) """

    return load

if __name__ == '__main__':
    indice = random_index()
    
    teste = demand_response_load(indice)
