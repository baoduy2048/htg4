import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import các module tự viết
from dss_engine import DecisionSupportSystem
from preprocessing import DataProcessor, post_process_cleaning

# Khởi tạo App
app = FastAPI()

# Config CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD RESOURCES (Chỉ load 1 lần khi server start) ---
model = None
outlier_remover = None
column_transformer = None

try:
    from catboost import CatBoostClassifier
    print("⏳ Đang tải model và các bộ xử lý...")
    
    # Load Model
    model = CatBoostClassifier()
    # Thử load model refined, nếu không có thì load v1
    try:
        model.load_model("credit_score_model_refined.cbm")
    except:
        model.load_model("credit_score_model_v1.cbm")

    # Load Transformers
    with open("OutlierRemover.pkl", "rb") as f:
        outlier_remover = pickle.load(f)
        
    with open("ColumnsTransformers.pkl", "rb") as f:
        column_transformer = pickle.load(f)
        
    print("✅ Đã tải thành công Model và Pickles!")
except Exception as e:
    print(f"❌ Lỗi nghiêm trọng: Không thể tải tài nguyên. Chi tiết: {e}")
    print("⚠️ Hãy chắc chắn bạn đã chạy 'python build_model.py' trước.")

# --- 2. DATA MODELS (Pydantic) ---

# Model con: Chỉ chứa thông tin người dùng (cho AI)
class UserProfile(BaseModel):
    age: int
    occupation: str 
    annual_income: float
    monthly_inhand_salary: float
    num_bank_accounts: int
    num_credit_card: int
    interest_rate: float
    num_of_loan: int
    type_of_loan: str
    delay_from_due_date: int
    num_of_delayed_payment: int
    changed_credit_limit: float
    num_credit_inquiries: int
    outstanding_debt: float
    credit_mix: str 
    credit_history_age: str # VD: "10 Years and 5 Months"
    payment_of_min_amount: str 
    total_emi_per_month: float
    amount_invested_monthly: float
    payment_behaviour: str 
    monthly_balance: float
    # Trường quan trọng đã bổ sung
    credit_utilization_ratio: float 

# Model cha: Chứa UserProfile + Biến môi trường (cho DSS)
class LoanApplicationRequest(BaseModel):
    user_profile: UserProfile   # Dữ liệu này sẽ vào AI
    
    # Các dữ liệu này sẽ vào DSS
    loan_amount: float          # M
    inflation_L: float          # Biến môi trường
    exchange_rate_D: float      # Biến môi trường
    geopolitics_W: float        # Biến môi trường
    reserve_V: float            # Biến môi trường

# --- 3. HELPER FUNCTIONS ---

def map_profile_to_dataframe(profile: UserProfile):
    """
    Chuyển đổi dữ liệu từ API (snake_case) sang DataFrame (Title_Case)
    để khớp 100% với định dạng mà Model đã được huấn luyện.
    """
    data_dict = {
        'Age': str(profile.age), # Convert sang string để khớp logic regex của DataProcessor
        'Occupation': profile.occupation,
        'Annual_Income': profile.annual_income,
        'Monthly_Inhand_Salary': profile.monthly_inhand_salary,
        'Num_Bank_Accounts': profile.num_bank_accounts,
        'Num_Credit_Card': profile.num_credit_card,
        'Interest_Rate': profile.interest_rate,
        'Num_of_Loan': profile.num_of_loan,
        'Type_of_Loan': profile.type_of_loan,
        'Delay_from_due_date': profile.delay_from_due_date,
        'Num_of_Delayed_Payment': profile.num_of_delayed_payment,
        'Changed_Credit_Limit': profile.changed_credit_limit,
        'Num_Credit_Inquiries': profile.num_credit_inquiries,
        'Credit_Mix': profile.credit_mix,
        'Outstanding_Debt': profile.outstanding_debt,
        'Credit_Utilization_Ratio': profile.credit_utilization_ratio, # Quan trọng
        'Credit_History_Age': profile.credit_history_age,
        'Payment_of_Min_Amount': profile.payment_of_min_amount,
        'Total_EMI_per_month': profile.total_emi_per_month,
        'Amount_invested_monthly': profile.amount_invested_monthly,
        'Payment_Behaviour': profile.payment_behaviour,
        'Monthly_Balance': profile.monthly_balance,
        
        # Các cột Dummy bắt buộc để Pipeline không lỗi (pipeline yêu cầu đủ cột)
        'Month': 'January',
        'Customer_ID': 'API_REQ',
        'Name': 'User',
        'SSN': '000',
        'ID': '000',
        'is_train': False
    }
    return pd.DataFrame([data_dict])

def get_credit_score_from_ai(profile: UserProfile):
    """Quy trình chạy qua AI Model"""
    if model is None:
        return 0.5 # Fallback an toàn

    # B1: Map dữ liệu
    input_df = map_profile_to_dataframe(profile)

    # B2: Preprocessing (Cleaning)
    processor = DataProcessor("Customer_ID", input_df)
    df_clean = processor.preprocess()
    df_clean = post_process_cleaning(df_clean)

    # B3: Loại bỏ cột rác
    cols_to_drop = ["Month", "Customer_ID", "Name", "SSN", "is_train", "ID"]
    df_model = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

    # B4: Outlier Removal (Dùng Transformer đã học)
    num_cols = df_model.select_dtypes(include="number").columns
    # Lưu ý: transform trả về array hoặc df tùy phiên bản, gán lại cho an toàn
    df_model[num_cols] = outlier_remover.transform(df_model[num_cols])

    # B5: Column Transformer (Scaling + Imputing)
    # Pipeline trả về numpy array, làm mất tên cột
    X_array = column_transformer.transform(df_model)
    
    # B6: Tái tạo DataFrame với tên cột (Bắt buộc cho CatBoost)
    feature_names = []
    # Lấy tên cột Categorical (SimpleImputer giữ nguyên cột)
    cat_cols_input = df_model.select_dtypes(exclude="number").columns.tolist()
    feature_names.extend(cat_cols_input)
    # Lấy tên cột Numerical
    num_cols_input = df_model.select_dtypes(include="number").columns.tolist()
    feature_names.extend(num_cols_input)
    
    X_final = pd.DataFrame(X_array, columns=feature_names)
    X_final = X_final.apply(pd.to_numeric, errors="ignore")

    # B7: Dự đoán
    label = model.predict(X_final)[0][0] # "Good", "Standard", "Poor"
    
    # B8: Mapping sang điểm số (Y)
    # Logic: Good -> Cao, Poor -> Thấp
    score_map = {
        "Good": 0.85,
        "Standard": 0.55,
        "Poor": 0.25
    }
    # Lấy thêm xác suất để làm score biến thiên tự nhiên hơn
    proba = model.predict_proba(X_final)[0]
    class_idx = list(model.classes_).index(label)
    confidence = proba[class_idx] # Độ tin cậy của model
    
    base_score = score_map.get(label, 0.5)
    
    # Điều chỉnh score một chút dựa trên độ tin cậy (optional)
    final_score = base_score * (0.9 + 0.2 * confidence) # Dao động nhẹ
    
    return min(max(final_score, 0.1), 0.99) # Clip trong khoảng 0.1 - 0.99

# --- 4. API ENDPOINT ---

@app.post("/api/evaluate")
def evaluate_loan(data: LoanApplicationRequest):
    print(f" Nhận yêu cầu vay: {data.loan_amount}")
    
    # --- BƯỚC 1: LẤY ĐIỂM TÍN DỤNG TỪ AI ---
    try:
        credit_score_Y = get_credit_score_from_ai(data.user_profile)
        print(f" AI Credit Score (Y): {credit_score_Y:.4f}")
    except Exception as e:
        print(f" Lỗi AI Model: {e}")
        # Trong trường hợp AI lỗi, có thể trả về lỗi hoặc dùng điểm mặc định
        credit_score_Y = 0.5 

    # --- BƯỚC 2: CHẠY HỆ THỐNG DSS ---
    # Các biến môi trường (L, D, W, V) được lấy từ request
    # và đưa thẳng vào DSS, không qua AI Model.
    
    constants = {
        'L_max': 10.0,
        'L_min': 0.0,
        'V_total': 100_000_000_000.0 # Ví dụ vốn tổng 100 tỷ
    }

    dss = DecisionSupportSystem()

    # Chuẩn hóa input cho phương trình DSS
    normalized_inputs = dss.normalize_inputs(
        Y=credit_score_Y,       # Kết quả từ AI
        M=data.loan_amount,     # Số tiền vay
        V=data.reserve_V,       # Vốn dự trữ (Admin nhập)
        L=data.inflation_L,     # Lạm phát (Admin nhập)
        D=data.exchange_rate_D, # Tỷ giá (Admin nhập)
        W=data.geopolitics_W,   # Chính trị (Admin nhập)
        constants=constants
    )

    # Tính toán kết quả cuối cùng (Z)
    result = dss.evaluate(normalized_inputs)
    
    # Bổ sung thông tin trả về
    result = dss.evaluate(normalized_inputs)
    
    # --- SỬA ĐOẠN NÀY ---
    # Đưa điểm tín dụng ra ngoài cùng để Frontend dễ hiển thị
    result["ai_credit_score"] = round(credit_score_Y, 2) 
    
    # Vẫn giữ details nếu cần debug
    result["details"] = {
        "ai_credit_score": round(credit_score_Y, 2),
        "input_inflation": data.inflation_L,
        "input_exchange": data.exchange_rate_D,
        "input_geopolitics": data.geopolitics_W
    }
    
    return result

# Chạy server (dùng cho debug trực tiếp)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
