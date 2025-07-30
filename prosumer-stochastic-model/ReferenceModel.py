# ---------- BIBLIOTECAS ----------#
from generate_scenarios import main as update_scenarios
import matplotlib.cm as cm
import sys
import time
import random as rnd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
import pyomo.pysp.util.rapper as rapper
from networkx.convert_matrix import _generate_weighted_edges
from pyomo.environ import *
import numpy as np
import json
import pathlib
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
# ---------- FIM BIBLIOTECAS ----------#

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

# ---------- INDEX ----------#
PRICES_INDEX = range(24*4)
TIME_INDEX = range(24*4)
# ---------- FIM INDEX ----------#

# ---------- LEITURA .JSON ----------#
path = str(pathlib.Path(__file__).parent.absolute())
with open(path + '/config.json', 'r') as f:
    CONFIG = json.load(f)
DEMAND = dict()
SPOT_PRICES = dict()
for i, j in CONFIG['scenarios'].items():
    data = {k: l for k, l in zip(TIME_INDEX, j['demand_data'])}
    DEMAND[i] = data
    data = {k: l for k, l in zip(TIME_INDEX, j['spot_price_data'])}
    SPOT_PRICES[i] = data
# ---------- FIM LEITURA .JSON ----------#

# ---------- DEFINIÇÃO DO MODELO ----------#
model = ConcreteModel(name="A")

# ---------- PARÂMETROS DO MODELO ----------#
# ---------- BATERIA ----------#

model.max_p_charge = Param(initialize=CONFIG['max_energy_flow'])
model.max_p_discharge = Param(initialize=CONFIG['max_energy_flow'])
model.storage_size = Param(initialize=CONFIG['storage_size'])
model.min_soc = Param(initialize=CONFIG['min_soc'])
model.max_soc = Param(initialize=CONFIG['max_soc'])

# ---------- MERCADO ----------#
model.bilateral_price = Param(initialize=CONFIG['bilateral_price'])
# Limite máximo do mercado bilateral
model.bilateral_max = Param(initialize=CONFIG['bilateral_max'])
model.max_spot_price_global = Param(initialize=CONFIG['max_spot_price'])

# ---------- RESPOSTA DA DEMANDA ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor',
#                    'LavadoraDePratos', 'ArCondicionado1', 'ArCondicionado2', 'ArCondicionado3', 'RoboAspirador'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0,
#                               'LavadoraDePratos': 0.3, 'ArCondicionado1': 1.5, 'ArCondicionado2': 2.0, 'ArCondicionado3': 1.5, 'RoboAspirador': 1}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4,
#                                     'LavadoraDePratos': 1*4, 'ArCondicionado1': 3*4, 'ArCondicionado2': 5*4, 'ArCondicionado3': 8*4, 'RoboAspirador': 3*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4,
#                                   'LavadoraDePratos': 18*4, 'ArCondicionado1': 10*4, 'ArCondicionado2': 8*4, 'ArCondicionado3': 2*4, 'RoboAspirador': 1*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4,
#                                    'LavadoraDePratos': 23*4, 'ArCondicionado1': 20*4, 'ArCondicionado2': 23*4, 'ArCondicionado3': 20*4, 'RoboAspirador': 22*4}, domain=NonNegativeIntegers)

# # ---------- CONFORTO ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4, 'Aquecedor': 6*4,
#                                                               'LavadoraDePratos': 21*4, 'ArCondicionado1': 16*4, 'ArCondicionado2': 17*4, 'ArCondicionado3': 2*4, 'RoboAspirador': 18*4}, domain=NonNegativeIntegers)

# # ---------- RESPOSTA DA DEMANDA 7 cargas ----------#
model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor',
                   'LavadoraDePratos', 'ArCondicionado1', 'ArCondicionado2', 'ArCondicionado3'])
model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0,
                              'LavadoraDePratos': 0.3, 'ArCondicionado1': 1.5, 'ArCondicionado2': 2.0, 'ArCondicionado3': 1.5}, domain=NonNegativeReals)
model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4,
                                    'LavadoraDePratos': 1*4, 'ArCondicionado1': 3*4, 'ArCondicionado2': 5*4, 'ArCondicionado3': 8*4}, domain=NonNegativeIntegers)
model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4,
                                  'LavadoraDePratos': 18*4, 'ArCondicionado1': 10*4, 'ArCondicionado2': 8*4, 'ArCondicionado3': 2*4}, domain=NonNegativeIntegers)
model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4,
                                   'LavadoraDePratos': 23*4, 'ArCondicionado1': 20*4, 'ArCondicionado2': 23*4, 'ArCondicionado3': 20*4}, domain=NonNegativeIntegers)

#---------- CONFORTO 7 cargas ----------#
model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4, 'Aquecedor': 6*4,
                                                              'LavadoraDePratos': 21*4, 'ArCondicionado1': 16*4, 'ArCondicionado2': 17*4, 'ArCondicionado3': 2*4}, domain=NonNegativeIntegers)

# ---------- RESPOSTA DA DEMANDA 6 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor',
#                    'LavadoraDePratos', 'ArCondicionado1', 'ArCondicionado2'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0,
#                               'LavadoraDePratos': 0.3, 'ArCondicionado1': 1.5, 'ArCondicionado2': 2.0}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4,
#                                     'LavadoraDePratos': 1*4, 'ArCondicionado1': 3*4, 'ArCondicionado2': 5*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4,
#                                   'LavadoraDePratos': 18*4, 'ArCondicionado1': 10*4, 'ArCondicionado2': 8*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4,
#                                    'LavadoraDePratos': 23*4, 'ArCondicionado1': 20*4, 'ArCondicionado2': 23*4}, domain=NonNegativeIntegers)

# #---------- CONFORTO 6 cargas ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4, 'Aquecedor': 6*4,
#                                                               'LavadoraDePratos': 21*4, 'ArCondicionado1': 16*4, 'ArCondicionado2': 17*4}, domain=NonNegativeIntegers)

# ---------- RESPOSTA DA DEMANDA 5 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor',
#                    'LavadoraDePratos', 'ArCondicionado1'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0,
#                               'LavadoraDePratos': 0.3, 'ArCondicionado1': 1.5}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4,
#                                     'LavadoraDePratos': 1*4, 'ArCondicionado1': 3*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4,
#                                   'LavadoraDePratos': 18*4, 'ArCondicionado1': 10*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4,
#                                    'LavadoraDePratos': 23*4, 'ArCondicionado1': 20*4}, domain=NonNegativeIntegers)

# #---------- CONFORTO 5 cargas ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4, 'Aquecedor': 6*4,
#                                                               'LavadoraDePratos': 21*4, 'ArCondicionado1': 16*4}, domain=NonNegativeIntegers)

# ---------- RESPOSTA DA DEMANDA REDUZIDO 4 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor','LavadoraDePratos'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0,
#                               'LavadoraDePratos': 0.3}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4,
#                                      'LavadoraDePratos': 1*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4,
#                                    'LavadoraDePratos': 18*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4,
#                                     'LavadoraDePratos': 23*4}, domain=NonNegativeIntegers)

# # # ---------- CONFORTO REDUZIDO 4 cargas ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 13*4, 'Aquecedor': 6*4,
#                                                                  'LavadoraDePratos': 18*4}, domain=NonNegativeIntegers)
# ---------- RESPOSTA DA DEMANDA REDUZIDO 3 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico', 'Aquecedor'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5, 'Aquecedor': 4.0}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4, 'Aquecedor': 0.25*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4, 'Aquecedor': 5*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4, 'Aquecedor': 7*4}, domain=NonNegativeIntegers)

# # ---------- CONFORTO REDUZIDO 3 cargas ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4, 'Aquecedor': 6*4}, domain=NonNegativeIntegers)

# ---------- RESPOSTA DA DEMANDA REDUZIDO 2 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar', 'FornoEletrico'])
# model.cargas_potencia = Param(model.cargas, initialize={'MaquinaDeLavar': 0.6, 'FornoEletrico': 1.5}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 1*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={'MaquinaDeLavar': 2*4, 'FornoEletrico': 10*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={'MaquinaDeLavar': 18*4, 'FornoEletrico': 16*4}, domain=NonNegativeIntegers)

# # # ---------- CONFORTO REDUZIDO 2 cargas ----------#
# model.cargas_tempo_desejado = Param(model.cargas, initialize={'MaquinaDeLavar': 8*4, 'FornoEletrico': 12*4}, domain=NonNegativeIntegers)

# # ---------- RESPOSTA DA DEMANDA REDUZIDO 1 cargas ----------#
# model.cargas = Set(initialize=['MaquinaDeLavar'])
# model.cargas_potencia = Param(model.cargas, initialize={
#                               'MaquinaDeLavar': 0.6}, domain=NonNegativeReals)
# model.cargas_horas_trabalho = Param(
#     model.cargas, initialize={'MaquinaDeLavar': 2*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_inicio = Param(model.cargas, initialize={
#                                   'MaquinaDeLavar': 2*4}, domain=NonNegativeIntegers)
# model.cargas_tempo_termino = Param(model.cargas, initialize={
#                                    'MaquinaDeLavar': 18*4}, domain=NonNegativeIntegers)

# # # # ---------- CONFORTO REDUZIDO 1 cargas ----------#
# model.cargas_tempo_desejado = Param(
#     model.cargas, initialize={'MaquinaDeLavar': 8*4}, domain=NonNegativeIntegers)

# ---------- PARÂMETROS COM ESTOCACIDADE ----------#
model.spot_prices = Param(PRICES_INDEX,
                          initialize={i: j for i, j in zip(
                              PRICES_INDEX, np.zeros(len(TIME_INDEX)))},
                          mutable=True)
model.demand = Param(TIME_INDEX,
                     initialize={i: j for i, j in zip(
                         TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                     mutable=True)
# ---------- FIM DOS PARÂMETROS DO MODELO ----------#

# ---------- VARIÁVEIS DO MODELO ----------#
# ---------- BATERIA ----------#
model.charging = Var(TIME_INDEX, within=Binary)
model.discharging = Var(TIME_INDEX, within=Binary)
model.p_charge = Var(TIME_INDEX,
                     initialize={i: j for i, j in zip(
                         TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                     domain=NonNegativeReals, bounds=(0.0, 2.0))
model.p_discharge = Var(TIME_INDEX,
                        initialize={i: j for i, j in zip(
                            TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                        domain=NonNegativeReals, bounds=(0.0, 2.0))
model.soc = Var(TIME_INDEX, domain=NonNegativeReals,
                bounds=(model.min_soc, model.max_soc))

# ---------- MERCADO ----------#
model.p_bilateral = Var(TIME_INDEX,
                        initialize={i: j for i, j in zip(
                            TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                        domain=NonNegativeReals,
                        bounds=(0.0, CONFIG['bilateral_max']))

model.p_spot = Var(TIME_INDEX,
                   initialize={i: j for i, j in zip(
                       TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                   domain=Reals)



# ---------- RESPOSTA DA DEMANDA ----------#
# # Variável binária, onde 1 significa que está ligada e 0 significa que está desligada em uma matriz de Cargas x Time_Index, para o caso em questão (8x96)
model.cargas_var_ligadas = Var(
    model.cargas, TIME_INDEX, domain=Binary, initialize=0)

# Variável binária, onde 1 significa o primeiro instante que "model.cargas_var_ligadas" assumiu o valor 1 em uma matriz de Cargas x Time_Index, para o caso em questão (8x96)
model.cargas_flag_partida = Var(
    model.cargas, TIME_INDEX, domain=Binary, initialize=0)

# Variável binária, onde 1 significa o primeiro instante que "model.cargas_var_ligadas" assumiu o valor 0 após "model.cargas_flag_partida" ter assumido o valor de 1 em uma matriz de Cargas x Time_Index, para o caso em questão (8x96)
model.cargas_flag_desliga = Var(
    model.cargas, TIME_INDEX, domain=Binary, initialize=0)

# # ---------- CONFORTO ----------#
model.diferenca_positiva = Var(
    model.cargas, domain=NonNegativeIntegers, initialize=0)
model.diferenca_negativa = Var(
    model.cargas, domain=NonNegativeIntegers, initialize=0)
model.tempo_inicio_real = Var(
    model.cargas, domain=NonNegativeIntegers, initialize=0)

# ---------- FIM DAS VARIÁVEIS DO MODELO ----------#

# ---------- EXPRESSÕES DO MODELO ----------#
# Expressão para somar a potência das cargas estocásticas com as cargas deslocáveis
# Demanda determinística recebe a soma em cada instante de tempo da multiplicação entre a potência de cada carga com a variável que determina se ela está ativa naquele momento.


def demanda_total_rule(model, t):
    # A demanda total é a soma da demanda externa e a potência das cargas ligadas no tempo t
    demanda_cargas = sum(
        model.cargas_potencia[a] * model.cargas_var_ligadas[a, t] for a in model.cargas)
    return model.demand[t] + demanda_cargas


# Adicionado
model.demanda_total = Expression(TIME_INDEX, rule=demanda_total_rule)


def ComputeStageCost_rule(model):
    return model.bilateral_price


model.StageCost = Expression(rule=ComputeStageCost_rule)

# Expressão que calcula o pior cenário do tempo de funcionamento de uma carga com conforto
def ponto_medio_janela_rule(model, a):
    """
    """
    return ((model.cargas_tempo_termino[a] - model.cargas_tempo_inicio[a]) - model.cargas_horas_trabalho[a])

# Cria uma expressão indexada por 'model.cargas'
model.ponto_medio_janela = Expression(model.cargas, rule=ponto_medio_janela_rule)


# Expressão que SOMA os pontos médios de TODAS as cargas.
def soma_pontos_medios_rule(model):
    """
    Soma os valores da expressão 'ponto_medio_janela' para todas as cargas.
    """
    return sum(model.ponto_medio_janela[a] for a in model.cargas)

# Cria uma única expressão com o valor da soma total
model.soma_pontos_medios = Expression(rule=soma_pontos_medios_rule)

# Na seção de Expressões

def custo_pior_cenario_spot_rule(model):
    # 1. Acessar o preço máximo através da nossa nova variável
    # Não usamos mais a função max() do Python.
    preco_spot_max = model.max_spot_price_global

    energia_total_consumida = (sum(model.cargas_potencia [a] * model.cargas_horas_trabalho[a] for a in model.cargas) + (sum(model.demand[t] for t in TIME_INDEX)))*0.25

    # 3. Multiplicar a energia total pelo preço máximo
    return (energia_total_consumida * preco_spot_max)

# A criação da expressão permanece a mesma
model.custo_pior_cenario_spot = Expression(rule=custo_pior_cenario_spot_rule)
# ---------- FIM DAS EXPRESSÕES DO MODELO ----------#


# ---------- FUNÇÃO OBJETIVO DO MODELO ----------#


def obj_rule(model):
    custo = dict()
    aux = list()
    a = 0
    dt = 0.25  # 15 min
    for t in TIME_INDEX:
        custo[t] = model.p_bilateral[t] * (model.bilateral_price) +\
            model.p_spot[t] * (model.spot_prices[t])
        aux.append(custo[t] * dt)
    aux = sum(aux)
    aux2 = aux/(model.custo_pior_cenario_spot)
    penalizacao_conforto = sum(
        model.diferenca_positiva[a] + model.diferenca_negativa[a] for a in model.cargas)
    desconforto = (penalizacao_conforto) /(model.soma_pontos_medios)
    return a*aux2+(1-a)*desconforto


model.obj = Objective(rule=obj_rule, sense=minimize)

# ---------- FIM DA FUNÇÃO OBJETIVO DO MODELO ----------#


# ---------- RESTRIÇÕES DO MODELO ----------#
# ---------- CONFORTO ----------#


def conforto_regra(model, a):
    return model.diferenca_positiva[a] - model.diferenca_negativa[a] == model.tempo_inicio_real[a] - model.cargas_tempo_desejado[a]


model.conforto_regra = Constraint(model.cargas, rule=conforto_regra)


def restricao_diferenca_positiva(model, a):
    return model.diferenca_positiva[a] >= 0


model.restricao_diferenca_positiva = Constraint(
    model.cargas, rule=restricao_diferenca_positiva)


def restricao_diferenca_negativa(model, a):
    return model.diferenca_negativa[a] >= 0


model.restricao_diferenca_negativa = Constraint(
    model.cargas, rule=restricao_diferenca_negativa)


def tempo_partida_real_constraint(model, a):
    tempo_inicio_real = sum(
        t * model.cargas_flag_partida[a, t] for t in TIME_INDEX)
    return model.tempo_inicio_real[a] == tempo_inicio_real


model.tempo_partida_real_constraint = Constraint(
    model.cargas, rule=tempo_partida_real_constraint)


# Função adicionada que não está sendo utilizada no código, mas pode ser útil futuramente
# def total_energy_balance_rule(model):
#     # A soma de todas as energias compradas no mercado bilateral e no mercado spot deve ser igual à soma da demanda total
#     soma_bilateral = sum(model.p_bilateral[t] for t in TIME_INDEX)
#     soma_spot = sum(model.p_spot[t] for t in TIME_INDEX)
#     soma_demanda_total = sum(model.demanda_total[t] for t in TIME_INDEX)
    
#     # A soma das energias deve ser igual à soma da demanda total
#     return soma_bilateral + soma_spot == soma_demanda_total

# # Adicionando a restrição ao modelo
# model.total_energy_balance_constraint = Constraint(rule=total_energy_balance_rule)

# ---------- FIM DO CONFORTO ----------#



# ---------- BALANÇO ENERGÉTICO E MERCADO ----------#
def balance_energy_constraint(model, t):
    # Energia comprada no mercado bilateral e spot
    energia_comprada = model.p_bilateral[t] + model.p_spot[t]

    # Demanda líquida (consumo - produção) ajustada pelo armazenamento de energia
    demanda_liquida = model.demanda_total[t] + \
        (model.p_charge[t] - model.p_discharge[t])

    # Caso a demanda líquida seja positiva (consumo maior que produção)
    if value(demanda_liquida) >= 0.0:
        # O total de energia comprada deve ser maior ou igual à demanda líquida
        return energia_comprada >= demanda_liquida
    else:
        # Caso a produção seja maior que o consumo (demanda líquida negativa), a energia do mercado spot pode ser usada para vender o excedente
        # O valor do mercado spot deve ser maior ou igual ao excedente
        return model.p_spot[t] >= demanda_liquida


model.balance_energy_constraint = Constraint(
    TIME_INDEX, rule=balance_energy_constraint)


def spot_energy_rule_constraint(model, t):
    # Demanda líquida ajustada pelo armazenamento de energia
    demanda_liquida = model.demanda_total[t] + \
        (model.p_charge[t] - model.p_discharge[t])

    # Caso a demanda líquida seja positiva (consumo maior que produção)
    if value(demanda_liquida) >= 0.0:
        # Não pode vender energia no mercado spot se a demanda líquida for positiva
        return model.p_spot[t] >= 0
    else:  # carga menor que producao
        return model.p_spot[t] <= demanda_liquida



model.spot_energy_rule_constraint = Constraint(
    TIME_INDEX, rule=spot_energy_rule_constraint)


# ---------- FIM DO BALANÇO ENERGÉTICO E MERCADO ----------#

# ---------- BATERIA ----------#


def charging_discharging_energy_constraint(model, t):
    return model.charging[t] + model.discharging[t] <= 1


model.charge_discharge_energy_constraint = Constraint(
    TIME_INDEX, rule=charging_discharging_energy_constraint)


def max_charge_rate_constraint(model, t):
    return model.p_charge[t] <= model.charging[t] * model.max_p_charge


model.max_charge_rate_constraint = Constraint(
    TIME_INDEX, rule=max_charge_rate_constraint)


def max_discharge_rate_constraint(model, t):
    return model.p_discharge[t] <= model.discharging[t] * model.max_p_discharge


model.max_discharge_rate_constraint = Constraint(
    TIME_INDEX, rule=max_discharge_rate_constraint)

model.init_soc_constraint = Constraint(
    expr=model.soc[0] == 0.2 * model.max_soc)

model.soc_memory_constraint = ConstraintList()
for t_m, t in zip(TIME_INDEX[:-1], TIME_INDEX[1:]):
    expr = model.soc[t_m] + \
        (model.p_charge[t_m] - model.p_discharge[t_m]) * 0.25
    model.soc_memory_constraint.add(model.soc[t] == expr)

charging = CONFIG.get('fixed_charging_states')
if charging is not None:
    model.charging_fixed_states_constraint = ConstraintList()
    for t in charging:
        model.charging_fixed_states_constraint.add(
            model.p_charge[t] == model.max_p_charge)

discharging = CONFIG.get('fixed_discharging_states')
if discharging is not None:
    model.discharging_fixed_states_constraint = ConstraintList()
    for t in discharging:
        model.discharging_fixed_states_constraint.add(
            model.p_discharge[t] == model.max_p_discharge)
# ---------- FIM DA BATERIA ----------#

# ---------- RESPOSTA DA DEMANDA ----------#
# ---------- CARGAS ININTERRUPTAS ----------#
# Restrição Implementada do H2V
# Restrições para as cargas deslocáveis funcionarem de forma ininterrupta
# Flag de partida e de desligamento não podem funcionar ao mesmo tempo


def flag_liga_desliga_restricao(model, a, t):
    return model.cargas_flag_partida[a, t] + model.cargas_flag_desliga[a, t] <= 1


model.flag_liga_desliga_restricao = Constraint(
    model.cargas, TIME_INDEX, rule=flag_liga_desliga_restricao)

# Restrição que relaciona a flag de partida e de desligamento com a variável "model.cargas_var_ligadas"


def estado_liga_desliga(model, a, t):
    return model.cargas_flag_partida[a, t] - model.cargas_flag_desliga[a, t] == model.cargas_var_ligadas[a, t] - model.cargas_var_ligadas[a, t-1]


model.estado_liga_desliga_restricao = Constraint(
    model.cargas, TIME_INDEX[1:], rule=estado_liga_desliga)


# Lista de restrição que relaciona as possibilidades de partida das cargas dentro do conjunto de tempo de trabalho que possui.
model.tempo_trabalho_cargas = ConstraintList()

for a in model.cargas:  # Iterando sobre todas as cargas
    tempo_inicio = int(model.cargas_tempo_inicio[a])
    # Converter para inteiro
    tempo_trabalho = int(model.cargas_horas_trabalho[a])
    tempo_termino = int(model.cargas_tempo_termino[a])
    janela_tempo = np.arange(tempo_inicio, tempo_termino - tempo_trabalho+2)

    for k in janela_tempo:
        estados = [model.cargas_var_ligadas[a, t]
                   for t in range(k, k + tempo_trabalho)]  # Usando 't'
        model.tempo_trabalho_cargas.add(
            sum(estados) >= tempo_trabalho * model.cargas_flag_partida[a, k])
# ---------- FIM DAS CARGAS ININTERRUPTAS ----------#

# ---------- TEMPO DE TRABALHO ----------#
# Quantidade de horas que cada carga deve trabalhar
# Restrição que atribui que a variável "model.cargas_var_ligadas" deve funcionar no tempo pré-estabelecido por "model.cargas_horas_trabalho[a]"
# Atenção: Verificar pra colocar o tempo minimo e tempo máximo da janela de trabalho


def carga_hora_trabalho_regra(model, a):
    return sum(model.cargas_var_ligadas[a, t] for t in TIME_INDEX) == (model.cargas_horas_trabalho[a])


model.carga_hora_trabalho_regra = Constraint(
    model.cargas, rule=carga_hora_trabalho_regra)
# ---------- FIM DO TEMPO DE TRABALHO ----------#

# ---------- JANELA DE TRABALHO ----------#
# Restrição que define a janela de tempo operacional de carga carga, onde a janela inicial é definida por "model.cargas_tempo_inicio" e a final por "model.cargas_tempo_termino".
# Sendo assim, a variável "model.cargas_var_ligadas" só pode assumir valores iguais a 1 dentro da janela de tempo pré-estabelecida.
# O tempo de funcionamento deve ser entre a janela inicial e final


def carga_tempo_restricao_regra(model, a, t):
    tempo_inicio = model.cargas_tempo_inicio[a]
    tempo_termino = model.cargas_tempo_termino[a]

    # Se o tempo t está dentro do intervalo de funcionamento da carga
    if tempo_inicio <= t <= tempo_termino:
        # A carga pode estar ligada
        # ou alguma outra condição, dependendo do seu modelo
        return model.cargas_var_ligadas[a, t] <= 1
    else:
        # Fora do intervalo, a carga não pode estar ligada
        return model.cargas_var_ligadas[a, t] == 0


model.cargas_tempo_restricao = Constraint(
    model.cargas, TIME_INDEX, rule=carga_tempo_restricao_regra)
# ---------- FIM DA JANELA DE TRABALHO ----------#


def pysp_instance_creation_callback(scenario_name, node_names):
    instance = model.clone()
    instance.demand.store_values(DEMAND[scenario_name])
    instance.spot_prices.store_values(SPOT_PRICES[scenario_name])
    instance.balance_energy_constraint.reconstruct()
    instance.spot_energy_rule_constraint.reconstruct()
    return instance