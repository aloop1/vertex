"""페널티 시스템 및 다목적 최적화 평가 로직"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from ga.config import ELEMENT_COST, ELEMENT_LIFE, ELEMENT_PHYSICS
except ImportError:
    print("데이터 임포트 불가")
    sys.exit(1)

try:
    from models.transformer_and_tree_ensemble import *
except ImportError as e:
    print(f"모델 구조 파일을 불러오지 못했습니다: {e}")
    sys.exit(1)

FINAL_FEATURES = [
    'stress', 'temp', 
    'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'W', 'Ni', 'Cu', 'V', 'Nb', 'N', 'Al', 'B', 'Co', 'Ta', 'O', 'Rh',
    'Ntemp', 'Ntime', 'Ttemp', 'Ttime', 'Atemp', 'Atime',
    'N_severity', 'T_severity', 'A_severity',
    'operating_severity', 'stress_temp_interaction', 'inverse_temp', 'total_ht_severity'
]

class FitnessEvaluator:
    def __init__(self):
        try:
            prep_data = joblib.load(os.path.join(parent_dir, 'data', 'preprocessor.pkl'))
            self.scaler = prep_data['scaler']
        except FileNotFoundError:
            print("preprocessor.pkl 누락")
            sys.exit(1)
            
        try:
            self.model = torch.load(os.path.join(parent_dir, 'models', 'best_model.pth'), map_location='cpu')
            self.model.eval()
            self.is_pytorch = True
        except Exception:
            self.model = joblib.load(os.path.join(parent_dir, 'models', 'best_model.pkl'))
            self.is_pytorch = False

    def _calculate_physics_features(self, df):
        for p in ["N", "T", "A"]:
            safe_time = np.maximum(df[f"{p}time"].astype(float), 1e-6)
            severity = df[f"{p}temp"].astype(float) * (20.0 + np.log10(safe_time))
            df[f"{p}_severity"] = np.where(df[f"{p}temp"] > 0, severity, 0.0)

        df['operating_severity'] = df['stress'] * df['temp']
        df['stress_temp_interaction'] = df['stress'] / df['temp']
        df['inverse_temp'] = 1.0 / df['temp']
        df['total_ht_severity'] = df['N_severity'] + df['T_severity'] + df['A_severity']
        return df

    def evaluate(self, individual, env_inputs):
        """
        individual: [19개 조성 + 6개 열처리] 유전자 리스트
        env_inputs: UI에서 입력받은 {stress, temp, target_life, oxidation, fluid_type, corrosion}
        """
        target_stress = env_inputs['stress']
        target_temp = env_inputs['temp']
        target_life = env_inputs['target_life']
        temp_c = target_temp - 273.15  # 섭씨 온도 변환
        
        # 1. 수명 예측을 위한 입력 전처리
        raw_data = [target_stress, target_temp] + list(individual)
        raw_cols = ['stress', 'temp'] + FINAL_FEATURES[2:27]
        df = pd.DataFrame([raw_data], columns=raw_cols)
        df = self._calculate_physics_features(df)
        
        scaled_x = self.scaler.transform(df[FINAL_FEATURES])
        
        if self.is_pytorch:
            with torch.no_grad():
                pred = self.model(torch.FloatTensor(scaled_x))
                base_score = float(pred.item())
        else:
            base_score = float(self.model.predict(scaled_x)[0])
# ======================================================


        # 2. 페널티 시스템
        penalty = 0.0
        
        # 유전자 리스트에서 각 원소 함량(wt%) 매핑
        comp = {name: individual[i] for i, name in enumerate(FINAL_FEATURES[2:21])}
        C=comp['C']; Si=comp['Si']; Mn=comp['Mn']; P=comp['P']; S=comp['S']
        Cr=comp['Cr']; Mo=comp['Mo']; W=comp['W']; Ni=comp['Ni']; Cu=comp['Cu']
        V=comp['V']; Nb=comp['Nb']; N=comp['N']; Al=comp['Al']; B=comp['B']
        Co=comp['Co']; Ta=comp['Ta']; O=comp['O']; Rh=comp['Rh']

        # [1] 용접성 제약
        # 1. 탄소당량 (CE_IIW)
        ce_iiw = C + (Mn/6.0) + ((Cr+Mo+V)/5.0) + ((Ni+Cu)/15.0)
        if ce_iiw > 0.45:
            penalty += (ce_iiw - 0.45) * 5.0

        # 2. 고온 균열 지수 (HCS)
        hcs_denominator = (3.0 * C + (Mn/10.0) + (Cr/15.0)) + 1e-6
        hcs_index = (S + P + (Si/25.0) + (Ni/100.0)) / hcs_denominator
        if hcs_index > 4.0:
            penalty += 2.0

        # 3. Mn/S 비율
        if S > 0.001 and (Mn / S) < 20.0:
            penalty += 1.5

        # [2] 산화 및 부식 제약
        # 1.PBR 약식 제어
        if env_inputs['oxidation'] != 'Vacuum':
            pbr_factor = (Cr * 2.0 + Al * 1.5) / (max(Si, 0.1) * 5.0) # 피막 형성 기여도
            if pbr_factor < 1.2 or pbr_factor > 2.3:
                penalty += 1.0

        # 2. 황화 저항성
        if env_inputs['fluid_type'] == 'Hydrocarbon+Sulfur':
            if Ni > 10.0 and Cr < (Ni * 1.5):
                penalty += 2.5

        # 3. 임계 공식 지수 (PREN)
        pren = Cr + 3.3 * Mo + 16 * N
        if env_inputs['corrosion'] and pren < 30.0:
            penalty += 1.2

        # [3] 위상학적 안정성 : 전자 공공수 계산 - 시그마상 페널티
        # PHACOM적용: 중량% -> 원자% 변환 후 평균 Nv 계산
        total_atomic_sum = 0
        element_moles = {}
        
        # 기지 금속(Fe) 계산 포함
        base_metal_wt = 100.0 - sum(comp.values())
        all_comp_with_fe = {**comp, 'Fe': max(base_metal_wt, 0)}
        
        for el, wt in all_comp_with_fe.items():
            atomic_wt = ELEMENT_PHYSICS[el][1]
            moles = wt / atomic_wt
            element_moles[el] = moles
            total_atomic_sum += moles
            
        avg_nv = 0
        for el, moles in element_moles.items():
            nv_val = ELEMENT_PHYSICS[el][0]
            avg_nv += (moles / total_atomic_sum) * nv_val
            
        if avg_nv > 2.52:
            penalty += (avg_nv - 2.52) * 10.0

        # [4] 공정 및 미량 원소 제약
        # 1. 붕소 과다
        if B > 0.01:
            penalty += 3.0

        # 2. 열처리 위계성 (Normalizing > Tempering > Annealing)
        Ntemp = individual[19]; Ttemp = individual[21]; Atemp = individual[23]
        if not (Ntemp > Ttemp >= Atemp):
            penalty += 5.0

        # 3. 목표 수명 미달 페널티 (도태시킴)
        target_log_life = np.log10(target_life)
        if base_score < target_log_life:
            penalty += (target_log_life - base_score) * 4.0
# ======================================================


        # 3. 최종 적합도 계산
        ELEMENT_LIFE = base_score - penalty
        
        # 4. 원가 계산 (ELEMENT_COST_PER_KG 활용)
        ELEMENT_COST = 0.0
        for idx, el_name in enumerate(element_names := ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'W', 'Ni', 'Cu', 'V', 'Nb', 'N', 'Al', 'B', 'Co', 'Ta', 'O', 'Rh']):
            weight_percent = individual[idx]
            ELEMENT_COST += (weight_percent / 100.0) * ELEMENT_COST[el_name]
        
        # 기지 금속(Fe) 비용 합산
        if base_metal_wt > 0:
            ELEMENT_COST += (base_metal_wt / 100.0) * ELEMENT_COST['Base']

        # 파레토 최적화를 위한 튜플 반환
        return (ELEMENT_LIFE, ELEMENT_COST)

evaluator = FitnessEvaluator()