from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import json
from feature_extractor import AMPFeatureExtractor
import warnings

warnings.filterwarnings('ignore')

# 初始化FastAPI应用
app = FastAPI(
    title="AntiMRSAPeptide.V1 API",
    description="基于ESM模型和集成学习的抗MRSA肽预测API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class SequenceRequest(BaseModel):
    sequences: List[str]


class PredictionResult(BaseModel):
    sequence: str
    probability: float
    prediction: str
    confidence: float
    details: Dict[str, Any]
    model_probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    overall_confidence: float
    processing_time: float
    feature_dimension: int


# 全局变量
scaler = None
models = {}
weights = {}
model_params = {}
feature_extractor = None


def load_models():
    """加载所有模型和权重"""
    global scaler, models, weights, model_params

    print("正在加载模型文件...")

    try:
        # 加载scaler
        scaler = joblib.load("models/final_scaler_no_leakage.pkl")
        print("✅ Scaler加载成功")

        # 加载权重
        weights_df = pd.read_csv("models/final_weights_no_leakage.csv")
        for _, row in weights_df.iterrows():
            weights[row['model']] = row['weight']
        print("✅ 权重加载成功")

        # 加载参数
        params_df = pd.read_csv("models/final_parameters_no_leakage.csv")
        model_params = params_df.set_index('model')['parameters'].to_dict()
        print("✅ 参数加载成功")

        # 加载所有模型
        model_files = {
            'xgboost': 'final_xgboost_model_no_leakage.pkl',
            'random_forest': 'final_random_forest_model_no_leakage.pkl',
            'lightgbm': 'final_lightgbm_model_no_leakage.pkl',
            'logistic_regression': 'final_logistic_regression_model_no_leakage.pkl',
            'svm': 'final_svm_model_no_leakage.pkl',
            'knn': 'final_knn_model_no_leakage.pkl'
        }

        for model_name, filename in model_files.items():
            try:
                model_path = f"models/{filename}"
                models[model_name] = joblib.load(model_path)
                print(f"✅ {model_name}模型加载成功")
            except Exception as e:
                print(f"⚠️  无法加载{model_name}模型: {str(e)}")

        print(f"✅ 共加载了{len(models)}个模型")

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise


def init_feature_extractor():
    """初始化特征提取器"""
    global feature_extractor
    print("正在初始化特征提取器...")
    try:
        feature_extractor = AMPFeatureExtractor()
        print("✅ 特征提取器初始化成功")
    except Exception as e:
        print(f"❌ 特征提取器初始化失败: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_models()
    init_feature_extractor()


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用AntiMRSAPeptide.V1 API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "model_weights": weights,
        "status": "运行中"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "scaler_loaded": scaler is not None,
        "extractor_ready": feature_extractor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SequenceRequest):
    """预测序列的抗MRSA活性"""
    import time
    start_time = time.time()

    try:
        # 1. 验证输入序列
        sequences = request.sequences
        if not sequences:
            raise HTTPException(status_code=400, detail="请输入至少一个序列")

        valid_sequences = []
        for seq in sequences:
            seq = seq.strip().upper()
            # 验证序列只包含有效氨基酸
            if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
                raise HTTPException(
                    status_code=400,
                    detail=f"序列包含无效氨基酸: {seq[:20]}..."
                )
            if len(seq) < 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"序列长度过短(至少5个氨基酸): {seq[:20]}..."
                )
            if len(seq) > 1000:
                raise HTTPException(
                    status_code=400,
                    detail=f"序列过长(最大1000个氨基酸): {seq[:20]}..."
                )
            valid_sequences.append(seq)

        # 2. 提取特征
        print(f"正在提取{len(valid_sequences)}个序列的特征...")
        features = feature_extractor.extract_sequence_features(valid_sequences)

        if features.size == 0:
            raise HTTPException(status_code=500, detail="特征提取失败")

        # 3. 标准化特征
        features_scaled = scaler.transform(features)

        # 4. 使用各个模型进行预测
        all_predictions = []
        model_predictions = {}

        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_scaled)[:, 1]
                else:
                    # 对于没有predict_proba的模型，使用predict
                    preds = model.predict(features_scaled)
                    probs = preds.astype(float)
                model_predictions[model_name] = probs
            except Exception as e:
                print(f"⚠️  {model_name}预测失败: {str(e)}")
                model_predictions[model_name] = np.zeros(len(valid_sequences))

        # 5. 计算加权平均
        weighted_predictions = np.zeros(len(valid_sequences))
        total_weight = 0

        for model_name, weight in weights.items():
            if model_name in model_predictions:
                weighted_predictions += weight * model_predictions[model_name]
                total_weight += weight

        if total_weight > 0:
            weighted_predictions /= total_weight

        # 6. 计算整体置信度（基于模型间的一致性）
        if len(model_predictions) > 1:
            all_probs = np.array(list(model_predictions.values()))
            confidence = 1.0 - np.std(all_probs, axis=0).mean()
        else:
            confidence = 0.8

        # 7. 准备响应数据
        predictions = []
        for i, (seq, prob) in enumerate(zip(valid_sequences, weighted_predictions)):
            # 收集各模型的预测概率
            model_probs = {}
            for model_name in model_predictions.keys():
                model_probs[model_name] = float(model_predictions[model_name][i])

            # 计算序列的物理化学特征
            seq_features = {
                "length": len(seq),
                "hydrophobic_ratio": (seq.count('A') + seq.count('V') + seq.count('L') +
                                      seq.count('I') + seq.count('M') + seq.count('F') +
                                      seq.count('W') + seq.count('P')) / len(seq),
                "positive_charge": (seq.count('R') + seq.count('K') + seq.count('H')) / len(seq),
                "negative_charge": (seq.count('D') + seq.count('E')) / len(seq),
                "aromatic_ratio": (seq.count('F') + seq.count('W') + seq.count('Y')) / len(seq)
            }

            prediction_result = PredictionResult(
                sequence=seq,
                probability=float(prob),
                prediction="抗MRSA肽" if prob >= 0.5 else "非抗MRSA肽",
                confidence=float(confidence),
                details=seq_features,
                model_probabilities=model_probs
            )
            predictions.append(prediction_result)

        processing_time = time.time() - start_time

        response = PredictionResponse(
            predictions=predictions,
            overall_confidence=float(confidence),
            processing_time=processing_time,
            feature_dimension=features.shape[1]
        )

        print(f"✅ 预测完成，处理时间: {processing_time:.2f}秒")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def get_models_info():
    """获取模型信息"""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            "type": type(model).__name__,
            "parameters": model_params.get(name, "N/A"),
            "weight": weights.get(name, 0.0)
        }

    return {
        "total_models": len(models),
        "models": model_info,
        "total_weights": sum(weights.values()),
        "feature_extractor": "ESM2_t6_8M_UR50D"
    }


@app.get("/system/info")
async def system_info():
    """获取系统信息"""
    import sys
    import torch
    import platform

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(feature_extractor.device) if feature_extractor else "N/A",
        "models_loaded": list(models.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)