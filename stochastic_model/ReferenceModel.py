from pprint import pprint
from pyomo.environ import *
import numpy as np
import json
import pathlib

import pyutilib.subprocess.GlobalData


# ----------------------------------
# index sets
# ----------------------------------
PRICES_INDEX = range(24 * 4)
TIME_INDEX = range(24 * 4)

# ----------------------------------
# DATA
# ----------------------------------
path = str(pathlib.Path(__file__).parent.absolute())
with open(path + '/config.json', 'r') as f:
    CONFIG = json.load(f)

DEMAND = dict()
SPOT_PRICES = dict()
tam_set_curt = []
tam_set_inte = []
tam_set_shift = []
tam_set_shift_duration = []
CURTAILABLE = {}
INTERRUPTIBLE = {}
SHIFTLABLE = {}
SHIFTLABLE_DURATION = {}

for i, j in CONFIG['scenarios'].items():
    data = {k: l for k, l in zip(TIME_INDEX, j['demand_data'])}
    DEMAND[i] = data

    data = {k: l for k, l in zip(TIME_INDEX, j['spot_price_data'])}
    SPOT_PRICES[i] = data

    if 'curtailable_load' in j:
        CURTAILABLE[i] = {}
        for n, m in enumerate(j['curtailable_load']):
            data = {(n, j): k for j, k in zip(TIME_INDEX, m)}
            CURTAILABLE[i].update(data)
        tam_set_curt.append(len(j['curtailable_load']))

    if 'interruptible_load' in j:
        INTERRUPTIBLE[i] = {}
        for n, m in enumerate(j['interruptible_load']):
            data = {(n, j): k for j, k in zip(TIME_INDEX, m)}
            INTERRUPTIBLE[i].update(data)
        tam_set_inte.append(len(j['interruptible_load']))
        
    if 'shiftable_load' in j:
        SHIFTLABLE[i] = {}
        for n, m in enumerate(j['shiftable_load']):
            data = {(n, j): k for j, k in zip(TIME_INDEX, m)}
            SHIFTLABLE[i].update(data)
        tam_set_shift.append(len(j['shiftable_load']))

set_curt = range(max(tam_set_curt))
set_inte = range(max(tam_set_inte))
set_shift = range(max(tam_set_shift))
set_max = range(max(max(tam_set_curt), max(tam_set_inte), max(tam_set_shift)))

# ----------------------------------
# model definition
# ----------------------------------
model = ConcreteModel(name="A")

# ----------------------------------
# parameters of the model
# ----------------------------------
if CONFIG['has_storage']:
    model.max_p_charge = Param(initialize=CONFIG['max_energy_flow'])
    model.max_p_discharge = Param(initialize=CONFIG['max_energy_flow'])
    model.storage_size = Param(initialize=CONFIG['storage_size'])
    model.min_soc = Param(initialize=CONFIG['min_soc'])
    model.max_soc = Param(initialize=CONFIG['max_soc'])

model.bilateral_price = Param(initialize=CONFIG['bilateral_price'])
model.prices = Param(PRICES_INDEX, initialize = lambda self, j: CONFIG['prices'][j]*100)
model.prices_curt = Param(set_curt,initialize = lambda self, j: CONFIG['prices_curt'][j] )
model.prices_inte = Param(set_inte,initialize = lambda self, j: CONFIG['prices_inte'][j] )
model.prices_shift = Param(set_shift,initialize = lambda self, j: CONFIG['prices_shift'][j] )
#model.prices = Param(PRICES_INDEX, initialize = lambda self, j: CONFIG['prices']*100)
# ----------------------------------
# parameters with stochasticity
# ----------------------------------
model.spot_prices = Param(PRICES_INDEX,
                          initialize={i: j for i, j in zip(PRICES_INDEX, np.zeros(len(TIME_INDEX)))},
                          mutable=True)

model.demand = Param(TIME_INDEX,
                     initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                     mutable=True)

model.P_curt = Param(set_curt, TIME_INDEX,
                     initialize={(i, j): k for i in set_curt
                                 for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                     mutable=True)

model.P_load = Param(set_inte, TIME_INDEX,
                     initialize={(i, j): k for i in set_inte
                                 for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                     mutable=True)

model.P_shift = Param(set_shift, TIME_INDEX,
                      initialize={(i, j): k for i in set_shift
                                  for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                      mutable=True)

model.T_shift = Param(set_shift, TIME_INDEX,
                      initialize={(i, j): k for i in set_shift
                              for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                      mutable=True)

# ----------------------------------
# variables of the model
# ----------------------------------
if CONFIG['has_storage']:
    model.charging = Var(TIME_INDEX, within=Binary)
    model.discharging = Var(TIME_INDEX, within=Binary)
    model.p_charge = Var(TIME_INDEX,
                         initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                         domain=NonNegativeReals)
    model.p_discharge = Var(TIME_INDEX,
                            initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                            domain=NonNegativeReals)
    model.soc = Var(TIME_INDEX, domain=NonNegativeReals, bounds=(model.min_soc, model.max_soc))

model.p_bilateral = Var(TIME_INDEX,
                        initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                        domain=NonNegativeReals,
                        bounds=(0.0, CONFIG['bilateral_max']))

model.p_spot = Var(TIME_INDEX,
                   initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                   domain=Reals)

model.demand_response = Var(TIME_INDEX,
                            initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                            domain=NonNegativeReals)

model.p_curt = Var(set_curt, TIME_INDEX,
                   initialize={(i, j): k for i in set_curt
                               for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                   domain=NonNegativeReals)

model.x_inte = Var(set_inte, TIME_INDEX, within=Binary)

model.p_inte = Var(set_inte, TIME_INDEX,
                   initialize={(i, j): k for i in set_inte
                               for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                   domain=NonNegativeReals)

model.total_load = Var(TIME_INDEX,
                       initialize={i: j for i, j in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                       domain=NonNegativeReals)

model.p_shift = Var(set_shift, TIME_INDEX,
                    initialize={(i, j): k for i in set_shift
                               for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                    domain=NonNegativeReals)

model.p_shift_neg = Var(set_shift, TIME_INDEX,
                        initialize={(i, j): k for i in set_shift
                               for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                        domain=NonNegativeReals)

model.p_shift_pos = Var(set_shift, TIME_INDEX,
                        initialize={(i, j): k for i in set_shift
                               for j, k in zip(TIME_INDEX, np.zeros(len(TIME_INDEX)))},
                        domain=NonNegativeReals)

# ----------------------------------
# objctive function model
# ----------------------------------
def obj_rule(model):
    custo = {}
    custo_curt = {}
    custo_inte = {}
    custo_shift = {}
    aux = []
    aux_curt = []
    aux_inte = []
    aux_shift = []
    dt = 0.25  # 15 min
    
    for l in set_max:
        for t in TIME_INDEX:
            if (l <= (max(tam_set_curt) - 1)) and (l <= (max(tam_set_inte) - 1)) and (l <= (max(tam_set_shift) - 1)):
                custo_curt[t] = model.p_curt[l,t] * model.prices[t]
                custo_inte[t] = model.p_inte[l,t] * model.prices[t]
                custo_shift[t] = model.p_shift_neg[l,t] * model.prices[t]
            elif (l <= (max(tam_set_curt) - 1)) and (l > (max(tam_set_inte) - 1)) and (l > (max(tam_set_shift) - 1)):
                custo_curt[t] = model.p_curt[l,t] * model.prices[t]
            elif (l > (max(tam_set_curt) - 1)) and (l <= (max(tam_set_inte) - 1)) and (l > (max(tam_set_shift) - 1)):
                custo_inte[t] = model.p_inte[l,t] * model.prices[t]
            else:
                custo_shift[t] = model.p_shift[l,t] * model.prices[t]

            aux_curt.append(custo_curt[t] * dt)
            aux_inte.append(custo_inte[t] * dt) 
            aux_shift.append(custo_shift[t] * dt)

    for t in TIME_INDEX:
        custo[t] = model.p_bilateral[t] * model.bilateral_price + \
                   model.p_spot[t] * model.spot_prices[t]
        aux.append(custo[t] * dt)

    aux_curt = (sum(aux_curt))
    aux_inte = (sum(aux_inte))
    aux_shift = (sum(aux_shift))
    aux = sum(aux)
    
    return aux + aux_curt + aux_inte + aux_shift


model.obj = Objective(rule=obj_rule, sense=minimize)

# ----------------------------------
# constraints of the model
# ----------------------------------
def balance_energy_constraint(model, t):
    aux1 = model.p_bilateral[t] + model.p_spot[t]

    if CONFIG['has_storage']:
        aux2 = model.demand[t] + (model.p_charge[t] - model.p_discharge[t]) + \
               (model.total_load[t] - model.demand_response[t])
    else:
        aux2 = model.demand[t] + (model.total_load[t] - model.demand_response[t])

    # a restrição definida aqui estabelece que:
    # 1) caso a demanda liquida seja positiva, ou seja, o consumo supere a producao,
    #    o prosumidor precisa garantir o balanço energético por meio da compra
    #    de energia no mercado bilateral e no mercado de tempo real.
    # 2) caso a demanda liquida seja negativa, ou seja a prosução supere o consumo,
    #    o prosumidor precisa garantir que o excesso de energia seja ou armazenado
    #    ou vendido no mercado de tempo real.
    if value(aux2) >= 0.0:  # consumo maior que producao
        return aux1 >= aux2
    else:  # carga menor que producao
        return model.p_spot[t] >= aux2


model.balance_energy_constraint = Constraint(TIME_INDEX, rule=balance_energy_constraint)


def spot_energy_rule_constraint(model, t):
    # a restrição definida aqui estabelece que:
    # não é possível ao prosumidor a venda de energia no mercado de tempo real
    # da energia que foi adquirida no mercado de contratação bilateral.
    if CONFIG['has_storage']:
        aux = model.demand[t] + (model.p_charge[t] - model.p_discharge[t]) + \
              (model.total_load[t] - model.demand_response[t])
    else:
        aux = model.demand[t] + (model.total_load[t] - model.demand_response[t])

    if value(aux) >= 0.0:  # consumo maior que producao
        return model.p_spot[t] >= 0
    else:  # carga menor que producao
        return model.p_spot[t] <= 0


model.spot_energy_rule_constraint = Constraint(TIME_INDEX, rule=spot_energy_rule_constraint)

if CONFIG['has_storage']:
    def charging_discharging_energy_constraint(model, t):
        return model.charging[t] + model.discharging[t] <= 1


    model.charge_discharge_energy_constraint = Constraint(TIME_INDEX, rule=charging_discharging_energy_constraint)


    def max_charge_rate_constraint(model, t):
        return model.p_charge[t] <= model.charging[t] * model.max_p_charge


    model.max_charge_rate_constraint = Constraint(TIME_INDEX, rule=max_charge_rate_constraint)


    def max_discharge_rate_constraint(model, t):
        return model.p_discharge[t] <= model.discharging[t] * model.max_p_discharge


    model.max_discharge_rate_constraint = Constraint(TIME_INDEX, rule=max_discharge_rate_constraint)

    model.init_soc_constraint = Constraint(expr=model.soc[0] == 0.2 * model.max_soc)

    model.soc_memory_constraint = ConstraintList()
    for t_m, t in zip(TIME_INDEX[:-1], TIME_INDEX[1:]):
        expr = model.soc[t_m] + (model.p_charge[t_m] - model.p_discharge[t_m]) * 0.25
        model.soc_memory_constraint.add(model.soc[t] == expr)

    charging = CONFIG.get('fixed_charging_states')
    if charging is not None:
        model.charging_fixed_states_constraint = ConstraintList()
        for t in charging:
            model.charging_fixed_states_constraint.add(model.p_charge[t] == model.max_p_charge)

    discharging = CONFIG.get('fixed_discharging_states')
    if discharging is not None:
        model.discharging_fixed_states_constraint = ConstraintList()
        for t in discharging:
            model.discharging_fixed_states_constraint.add(model.p_discharge[t] == model.max_p_discharge)


model.curtailable_load = Constraint(set_curt, TIME_INDEX, rule = lambda model,  j, i: model.p_curt[j,i] <= model.P_curt[j,i])
model.interruptible_load = Constraint(set_inte, TIME_INDEX, rule = lambda model,  j, i: model.p_inte[j,i] == model.P_load[j,i] * model.x_inte[j,i])
model.shiftable_load = ConstraintList()

for l in set_shift:
    aux1 = []
    aux2 = []
    while True:
        for t in TIME_INDEX:
            model.shiftable_load.add(model.p_shift[l,t] <= model.P_shift[l,t] + model.P_shift[l,t] * 0.2)
            model.shiftable_load.add(model.p_shift[l,t] >= model.P_shift[l,t] - model.P_shift[l,t] * 0.2)
            model.shiftable_load.add(model.p_shift_pos[l,t] >= 0)
            model.shiftable_load.add(model.p_shift_neg[l,t] >= 0)
            model.shiftable_load.add(model.P_shift[l,t] + model.p_shift_pos[l,t] == model.p_shift[l,t] + model.p_shift_neg[l,t])

            aux1.append(model.p_shift[l,t])
            aux2.append(model.P_shift[l,t])
        
        aux1 = sum(aux1)
        aux2 = sum(aux2)

        if aux1 == aux2:
            break
        

def demand_total_load_constraint(model, t):
    aux = []
    for l in set_max:
        if (l <= max(tam_set_curt) - 1) and (l <= max(tam_set_inte) - 1) and (l <= max(tam_set_shift) - 1):
            aux.append(model.P_curt[l,t] + model.P_load[l,t] + model.P_shift[l,t])
        elif (l <= max(tam_set_curt) - 1) and (l > max(tam_set_inte) - 1) and (l > max(tam_set_shift) - 1):
            aux.append(model.P_curt[l,t])
        elif (l > max(tam_set_curt) - 1) and (l <= max(tam_set_inte) - 1) and (l > max(tam_set_shift) - 1):
            aux.append(model.P_load[l,t])
        else:
            aux.append(model.P_shift[l,t])
        
    return model.total_load[t] == sum(aux)


model.load_constraint = Constraint(TIME_INDEX, rule = demand_total_load_constraint)


def demand_balance_constraint(model, t):
    aux = []
    for l in set_max:
        if (l <= max(tam_set_curt) - 1) and (l <= max(tam_set_inte) - 1) and (l <= max(tam_set_shift) - 1):
            aux.append(model.p_curt[(l, t)] + model.p_inte[(l, t)] + model.p_shift[l,t])
        elif (l <= max(tam_set_curt) - 1) and (l > max(tam_set_inte) - 1) and (l > max(tam_set_shift) - 1):
            aux.append(model.p_curt[(l, t)])
        elif (l > max(tam_set_curt) - 1) and (l <= max(tam_set_inte) - 1) and (l > max(tam_set_shift) - 1):
            aux.append(model.p_inte[(l, t)])
        else:
            aux.append(model.p_shift[l,t])

    return model.demand_response[t] == sum(aux)
    

model.demand_balance = Constraint(TIME_INDEX, rule = demand_balance_constraint)

#teste = time_interrupt(CONFIG['prices_DSO'], TIME_INDEX)

# -----------------------------------
# Stage-specific cost computations
# -----------------------------------

def ComputeStageCost_rule(model):
    return model.bilateral_price


model.StageCost = Expression(rule=ComputeStageCost_rule)


# ----------------------------------
# Solving the model
# ----------------------------------
# solver = SolverFactory('cplex')
# results = solver.solve(model)
# if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
#     print ("this is feasible and optimal")
# elif results.solver.termination_condition == TerminationCondition.infeasible:
#     print ("do something about it? or exit?")
# else:
#     # something else is wrong
#     print (str(results.solver))


def pysp_instance_creation_callback(scenario_name, node_names):
    instance = model.clone()
    instance.demand.store_values(DEMAND[scenario_name])
    instance.spot_prices.store_values(SPOT_PRICES[scenario_name])
    instance.P_curt.store_values(CURTAILABLE[scenario_name])
    instance.P_load.store_values(INTERRUPTIBLE[scenario_name])
    instance.P_shift.store_values(SHIFTLABLE[scenario_name])
    instance.balance_energy_constraint.reconstruct()
    instance.spot_energy_rule_constraint.reconstruct()
    #instance.load_constraint.reconstruct()
    #instance.demand_balance.reconstruct()
    return instance
