import numpy as np

class DecisionSupportSystem:
    def __init__(self):
        # Trọng số AHP (Thứ tự: Y, rM, rL, rW, rV)
        self.weights_ahp = np.array([0.42, 0.26, 0.13, 0.07, 0.12])

        # Trọng số OWA (Thận trọng - Tăng dần áp lực cho rủi ro)
        # Logic: Sắp xếp input giảm dần, rồi nhân với trọng số này.
        # Giá trị nhỏ nhất (rủi ro nhất) sẽ nhân với 0.42 (trọng số lớn nhất)
        self.weights_owa = np.array([0.07, 0.12, 0.13, 0.26, 0.42])

        # Ngưỡng hệ thống
        self.ALPHA = 0.5      # Ngưỡng sàn T(S)
        self.THETA = 0.3      # rM tối thiểu
        self.GAMMA = 0.5      # E tối thiểu

    def normalize_inputs(self, Y, M, V, L, D, W, constants):
        """Chuẩn hóa các biến về miền [0, 1] - [cite: 681-696]"""
        # Y (Credit Score) đã ở dạng
        
        # rL: Lạm phát (Càng cao càng tệ)
        rL = (constants['L_max'] - L) / (constants['L_max'] - constants['L_min'])
        
        # rD: Tỷ giá (Biến động cao là rủi ro, giả sử D là variance đã chuẩn hóa)
        rD = 1 - D 
        
        # rW: Địa chính trị (1 là chiến tranh, 0 là hòa bình -> rW cần đảo ngược)
        rW = 1 - W
        
        # rV: Vốn dự trữ (Càng nhiều càng tốt)
        rV = V / constants['V_total']
        
        # rM: Quy mô khoản vay (M càng lớn so với V thì rM càng nhỏ -> rủi ro cao)
        rM = 1 - (M / V)

        return {
            "Y": np.clip(Y, 0, 1),
            "rM": np.clip(rM, 0, 1),
            "rL": np.clip(rL, 0, 1),
            "rW": np.clip(rW, 0, 1),
            "rV": np.clip(rV, 0, 1)
        }

    def evaluate(self, inputs):
        # Vector input theo thứ tự trọng số AHP
        input_vector = np.array([inputs['Y'], inputs['rM'], inputs['rL'], inputs['rW'], inputs['rV']])

        # 1. Tính Rủi ro tổng hợp (E)
        E = np.dot(self.weights_ahp, input_vector)

        # 2. Tính Chỉ số Môi trường & Hệ thống (S) dùng OWA
        # Sắp xếp giảm dần (Best -> Worst)
        sorted_inputs = np.sort(input_vector)[::-1]
        S = np.dot(self.weights_owa, sorted_inputs)

        # 3. Tính Ngưỡng thích ứng T(S)
        T_S = self.ALPHA + (1 - self.ALPHA) * (1 - S)

        # 4. Logic quyết định (Phương pháp HỘI)
        cond1 = inputs['Y'] >= T_S      # Điểm tín dụng >= Ngưỡng thích ứng
        cond2 = inputs['rM'] >= self.THETA # Khoản vay không quá lớn
        cond3 = E >= self.GAMMA         # Rủi ro tổng hợp an toàn

        return {
            "is_approved": bool(cond1 and cond2 and cond3),
            "metrics": {
                "E_score": round(E, 4),
                "S_score": round(S, 4),
                "Threshold_TS": round(T_S, 4),
                "normalized_vars": inputs
            },
            "conditions": {
                "Credit_Score_Check": bool(cond1),
                "Loan_Amount_Check": bool(cond2),
                "Risk_Score_Check": bool(cond3)
            }
        }
