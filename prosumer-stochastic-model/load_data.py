"""
Este arquivo carrega perfis de carga e de geração PV para serem utilizadas pelos prosumers
"""
# Carrega as bibliotecas
import pandas as pd
import numpy as np
import random
import os

path = os.path.dirname(os.path.abspath(__file__))
# Carrega as informações da rede e do mercado e aloca nas variáveis Grid_Data_Path e Market_Data_Path
GRID_DATA_PATH = path + '/1-MVLV-urban-5.303-1-no_sw'
MARKET_DATA_PATH = path + '/market-data'
# Define que serão 30 dias
DAYS = 30

# data frame com as coordenadas da rede
coordinates_df = pd.read_csv('{}/Coordinates.csv'.format(GRID_DATA_PATH), delimiter=';') # data frame com as linhas da rede de distribuicao
line_df = pd.read_csv('{}/Line.csv'.format(GRID_DATA_PATH), delimiter=';') # data frame com os nós da rede de distribuição
node_df = pd.read_csv('{}/Node.csv'.format(GRID_DATA_PATH), delimiter=';') # data frame com os tipos de linhas da rede de distribuição
line_type = pd.read_csv('{}/LineType.csv'.format(GRID_DATA_PATH), delimiter=';') # data frame com cargas em cada nó da rede de distribuição
load_df = pd.read_csv('{}/Load.csv'.format(GRID_DATA_PATH), delimiter=';')[138:242] # perfis de carga associados aos nodes da rede
load_profile_df = pd.read_csv('{}/LoadProfile.csv'.format(GRID_DATA_PATH), delimiter=';') # data frame com geradores em cada nó da rede de distribuição
gen_df = pd.read_csv('{}/RES.csv'.format(GRID_DATA_PATH), delimiter=';')[134:] # perfis de geração associados aos nós da rede
gen_profile_df = pd.read_csv('{}/RESProfile.csv'.format(GRID_DATA_PATH), delimiter=';') # Data frame com os preços do mercado spot
spot_prices_df = pd.read_excel('{}/Nordpool_Market_Data-3.xlsx'.format(MARKET_DATA_PATH))

# print (coordinates_df)
# print (line_df)
# print (node_df)
# print (line_type)
# print (load_df)
# print (load_profile_df)
# print (gen_df)
# print (gen_profile_df)
# print (spot_prices_df)

def get_load_data(load_index=None):
# Cria a variável month_delta que recebe a quantidade de 15minutos existem em um mês
    month_delta = 24 * 4 * 30 * 1
    #print ("month_delta:", month_delta)
    if load_index != None:
        load = load_df.iloc[load_index % 100]
        # start_day recebe metade de um mês
        #print ("load:", load)
        start_day = 15 * 24 * 4 + month_delta
        #print ("start_day:",start_day)
    else:
        load_index = random.randint(0, len(load_df) - 1)
        #print ("load_index:",load_index)
        load = load_df.iloc[load_index]
        #print ("load:",load)
        start_day = random.randint(0, 30) * 24 * 4 + month_delta
        #print ("start_day:",start_day)
    # Fim de um dia é um mês para outro
    end_day = start_day + 30 * 24 * 4
    #print ("end_day:",end_day)
    # load_p_data recebe os valores de _pload que estão em Load.csv compreendido na faixa de um dia.
    # load_q_data recebe os valores de _qload que estão em Load.csv compreendido na faixa de um dia.
    load_p_data = load_profile_df['{}_pload'.format(load.profile)][start_day:end_day]
    load_q_data = load_profile_df['{}_qload'.format(load.profile)][start_day:end_day]
    #print ("load_p_data:",load_p_data)
    #print ("load_q_data:",load_q_data)
    # Recebe os valores de load_p_data e transforma em uma Series com um range de tempo de um mês
    # Recebe os valores de load_q_data e transforma em uma Series com um range de tempo de um mês
    load_p_data = pd.Series(load_p_data.values, index=range(24 * 4 * DAYS))
    load_q_data = pd.Series(load_q_data.values, index=range(24 * 4 * DAYS))
    #print ("load_p_data:",load_p_data)
    #print ("load_q_data:",load_q_data)
    # Loads_df recebe a concatenação entre p_data q_data e associa horizontalmente
    loads_df = pd.concat([load_p_data, load_q_data], axis=1)
    #print ("loads_df:", loads_df)
    # Adicionada os índices das colunas com o valor de pload e qload
    loads_df.columns = ['pload', 'qload']
    #print ("loads_df.columns:", loads_df.columns)
    # Retorna as cargas p_load e q_load em uma faixa de um mês
    return loads_df


def get_gen_data(gen_index=None):
    # Cria a variável month_delta que recebe a quantidade de 15minutos existem em um mês
    month_delta = 24 * 4 * 30 * 1
    #print("month_delta", month_delta)
    if gen_index != None:
        gen = gen_df.iloc[gen_index % 12]
        #print("gen:", gen)
        # start_day recebe metade de um mês
        start_day = 15 * 24 * 4 + month_delta
        #print("start_day", start_day)
    else:
        gen_index = random.randint(0, len(gen_df) - 1)
        #print("gen_index", gen_index)
        gen = gen_df.iloc[gen_index]
        #print("gen:", gen)
        start_day = random.randint(0, 30) * 24 * 4 + month_delta
        #print("start_day", start_day)

    end_day = start_day + 30 * 24 * 4
    #print("end_day", end_day)
    gen_p_data = gen_profile_df[gen.profile][start_day:end_day]
    #print("gen_p_data", gen_p_data)
    gen_p_data = pd.Series(gen_p_data.values, index=range(24 * 4 * DAYS))
    #print("gen_p_data", gen_p_data)
    gens_df = pd.DataFrame(gen_p_data, columns=['pgen'])
    #print("gens_df", gens_df)
    return gens_df


def get_spot_prices_data(spot_index=None):

    if spot_index:
        month_delta = 24 * 30 * 6
        #print("month_delta", month_delta)
        start_day = (spot_index % 30) * 24 + month_delta
        #print("start_day", start_day)
        end_day = start_day + 30 * 24
        #print("end_day", end_day)
    else:
        month_delta = 24 * 30 * 6
        # print("month_delta", month_delta)
        start_day = random.randint(0, 30) * 24 + month_delta
        # print("start_day", start_day)
        end_day = start_day + 30 * 24
        # print("end_day", end_day)

    spot_p_data = spot_prices_df.SpotPriceEUR[start_day:end_day]
    # print("spot_p_data", spot_p_data)

    # converte os dados de preços da base horária para base de 15min
    aux = np.array([])
    for i in spot_p_data:
        aux = np.concatenate((aux, np.ones(4) * i))
        #print("aux", aux)

    spot_p_data = pd.Series(aux, index=range(24 * 4 * DAYS))
    # print("spot_p_data", spot_p_data)
    spot_df = pd.DataFrame(spot_p_data/1000, columns=['Price'])
    # print("spot_df", spot_df)
    return spot_df

if __name__ == '__main__':
    # LOAD DATA
    loads = list()
    #print ("loads", loads)
    for load_idx in range(10):
        loads.append(get_load_data(random.uniform(3.0, 5.0)))
    # PLOT DATA
    for l in loads:
        l.pload[:2 * 24 * 4].plot()

    sum(loads).pload[:2 * 24 * 4].plot()
