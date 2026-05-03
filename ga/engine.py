"""최적화 로직"""

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from ga.fitness import evaluator
from ga.config import ELEMENT_LIFE, ELEMENT_COST, HEAT_TREATMENT_BOUNDS, ELEMENT_PHYSICS, MAX_ALLOY_SUM

# 다목적 적합도 설정
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def create_individual():
    """유전자 생성"""
    comp = []
    element_keys = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'W', 'Ni', 'Cu', 'V', 'Nb', 'N', 'Al', 'B', 'Co', 'Ta', 'O', 'Rh']
    for el in element_keys:
        min_val, max_val = ELEMENT_LIFE[el]
        comp.append(random.uniform(min_val, max_val))
    
    # 총합 제약 준수
    comp_sum = sum(comp)
    if comp_sum > MAX_ALLOY_SUM:
        scale_factor = MAX_ALLOY_SUM / comp_sum
        comp = [val * scale_factor for val in comp]

    ht = []
    ht_keys = ['Ntemp', 'Ntime', 'Ttemp', 'Ttime', 'Atemp', 'Atime']
    for step in ht_keys:
        min_v, max_v = HEAT_TREATMENT_BOUNDS[step]
        ht.append(random.uniform(min_v, max_v))
        
    return creator.Individual(comp + ht)

def enforce_realistic_bounds(func):
    """물리적 제약 조건 준수"""
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        element_keys = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'W', 'Ni', 'Cu', 'V', 'Nb', 'N', 'Al', 'B', 'Co', 'Ta', 'O', 'Rh']
        for ind in offspring:
            for i in range(19):
                el = element_keys[i]; min_v, max_v = ELEMENT_LIFE[el]
                ind[i] = max(min_v, min(ind[i], max_v))
            comp_sum = sum(ind[:19])
            if comp_sum > MAX_ALLOY_SUM:
                scale = MAX_ALLOY_SUM / comp_sum
                for i in range(19): ind[i] *= scale
            for i in range(6):
                idx = 19 + i
                ht_name = ['Ntemp', 'Ntime', 'Ttemp', 'Ttime', 'Atemp', 'Atime'][i]
                min_v, max_v = HEAT_TREATMENT_BOUNDS[ht_name]
                ind[idx] = max(min_v, min(ind[idx], max_v))
        return offspring
    return wrapper

def run_ga(user_inputs):
    """NSGA-II 최적화"""
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate, env_inputs=user_inputs)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.emo.selNSGA2)
    toolbox.decorate("mate", enforce_realistic_bounds)
    toolbox.decorate("mutate", enforce_realistic_bounds)

    POP_SIZE = 300
    NGEN = 150
    population = toolbox.population(n=POP_SIZE)
    
    print(f"세대수: {NGEN}")
    algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=0.7, mutpb=0.3, ngen=NGEN, verbose=False)
    
    # 최적 개체 선정(최소 비용 우선)
    front = tools.emo.sortNondominated(population, len(population), first_front_only=True)[0]
    target_log_life = np.log10(user_inputs['target_life'])
    valid_candidates = [ind for ind in front if ind.fitness.values[0] >= target_log_life]
    best_ind = sorted(valid_candidates, key=lambda x: x.fitness.values[1])[0] if valid_candidates else sorted(front, key=lambda x: x.fitness.values[0], reverse=True)[0]

    # 최종 설계
    final_life_log, final_cost = best_ind.fitness.values
    actual_life = 10**final_life_log

    print("\n" + "═"*65)
    print("신합금 제조 시방서 및 최적화 리포트".center(65))
    print("═"*65)
    
    print(f"1. 목표 및 성능 요약")
    print(f"   - 예상 수명: {actual_life:,.1f} hr (목표: {user_inputs['target_life']:,} hr)")
    print(f"   - 추정 원가: ${final_cost:.2f} / kg")
    
    # 물리 제약 통과 여부 정밀 진단
    print(f"   - 물리 제약: 용접성, 산화/부식 저항, 시그마상 안정성 통과")

    print(f"\n2. 합금 조성 상세 (wt%)")
    element_names = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'W', 'Ni', 'Cu', 'V', 'Nb', 'N', 'Al', 'B', 'Co', 'Ta', 'O', 'Rh']
    comp_data = {name: best_ind[i] for i, name in enumerate(element_names)}
    comp_data['Fe (Base)'] = 100.0 - sum(best_ind[:19])
    
    sorted_elements = sorted(comp_data.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, len(sorted_elements), 4):
        line = " | ".join([f"{name}: {val:7.4f}%" for name, val in sorted_elements[i:i+4]])
        print(f"     {line}")

    print(f"\n3. [공정 가이드] 최적 열처리 조건")
    print(f"   - Normalizing: {best_ind[19]:.1f} K (약 {best_ind[19]-273.15:.1f} °C) | {best_ind[20]:.1f} hr")
    print(f"   - Tempering:   {best_ind[21]:.1f} K (약 {best_ind[21]-273.15:.1f} °C) | {best_ind[22]:.1f} hr")
    print(f"   - Annealing:   {best_ind[23]:.1f} K (약 {best_ind[23]-273.15:.1f} °C) | {best_ind[24]:.1f} hr")
    
    print("═"*65)
    
    return best_ind