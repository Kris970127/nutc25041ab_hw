import os
import uuid
import time
import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# === 基礎設定 ===
STUDENT_ID = "1111232041"
client = QdrantClient(url="http://localhost:6333")
EMBED_API = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"

os.makedirs("day5", exist_ok=True)

def get_embedding(texts):
    """獲取 Embedding，加入自動重試機制以防 502/Timeout"""
    payload = {"texts": texts, "task_description": "檢索技術文件", "normalize": True}
    for i in range(5): # 增加重試次數
        try:
            res = requests.post(EMBED_API, json=payload, timeout=25)
            if res.status_code == 200:
                return res.json()['embeddings']
        except:
            time.sleep(3)
    return [[0] * 4096]

def get_eval_score(q_id, retrieved_text):
    """評分 API，加入自動重試機制"""
    payload = {"q_id": int(q_id), "student_answer": str(retrieved_text)}
    for i in range(5):
        try:
            res = requests.post(SUBMIT_URL, json=payload, timeout=25)
            if res.status_code == 200:
                return float(res.json().get('score', 0.0))
        except:
            time.sleep(3)
    return 0.0

def get_chunks(method, text):
    if method == "固定大小":
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="")
    elif method == "滑動視窗":
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    elif method == "語意切塊":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=["\n\n", "。", "！", "？"])
    return [doc.page_content for doc in splitter.create_documents([text])]

def run_evaluation():
    data_files = ["data_01.txt", "data_02.txt", "data_03.txt", "data_04.txt", "data_05.txt"]
    questions_df = pd.read_csv("questions.csv")
    methods = ["固定大小", "滑動視窗", "語意切塊"]
    
    # 偵測維度
    sample_vec = get_embedding(["測試"])[0]
    vector_dim = len(sample_vec)
    print(f"✅ 偵測維度: {vector_dim}")

    final_results = []
    summary_data = []

    for m in methods:
        print(f"\n>>> 正在執行方法：{m}")
        col_name = f"hw5_{STUDENT_ID}_{uuid.uuid4().hex[:4]}"
        client.create_collection(col_name, VectorParams(size=vector_dim, distance=Distance.COSINE))

        # 1. 建立索引
        all_p = []
        for f_name in data_files:
            with open(f_name, "r", encoding="utf-8") as f:
                chunks = get_chunks(m, f.read())
                for c in chunks:
                    all_p.append({"text": c, "source": f_name})
        
        texts = [p['text'] for p in all_p]
        vectors = get_embedding(texts)
        points = [PointStruct(id=uuid.uuid4().hex, vector=vectors[i], payload=all_p[i]) for i in range(len(vectors))]
        client.upsert(col_name, points=points)

        # 2. 檢索與評分
        method_scores = []
        for _, row in questions_df.iterrows():
            q_text = row['questions']
            q_vec = get_embedding([q_text])[0]
            search = client.query_points(col_name, query=q_vec, limit=1).points
            
            if search:
                hit = search[0]
                score = get_eval_score(row['q_id'], hit.payload['text'])
                method_scores.append(score)
                print(f"   Q{row['q_id']} 得分: {score:.4f}")
                
                final_results.append({
                    "id": uuid.uuid4().hex,
                    "q_id": row['q_id'],
                    "method": m,
                    "retrieve_text": hit.payload['text'],
                    "score": score,
                    "source": hit.payload['source']
                })
        
        # 3. 計算該方法的平均分
        avg_score = sum(method_scores) / len(method_scores) if method_scores else 0
        summary_data.append({"方法": m, "平均分數": f"{avg_score:.4f}"})
        client.delete_collection(col_name)

    # 產出主 CSV
    df = pd.DataFrame(final_results)
    output_path = f"day5/{STUDENT_ID}_RAG_HW_01.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 顯示平均分總表
    print("\n" + "="*30)
    print(f"學號 {STUDENT_ID} 評測結果總覽")
    print("-" * 30)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*30)
    print(f"🎉 成功跑完 60 筆資料！CSV 已產出。")

if __name__ == "__main__":
    run_evaluation()