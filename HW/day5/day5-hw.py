import os
import uuid
import pandas as pd
import requests
import time
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# === 0. 配置與初始化 ===
STUDENT_ID = "1111232041"
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

client = QdrantClient(url="http://localhost:6333")

class CustomEmbeddings:
    def embed_documents(self, texts): return get_embeddings(texts)
    def embed_query(self, text): return get_embeddings([text])[0]

def get_embeddings(texts):
    if not texts: return []
    payload = {"texts": texts, "normalize": True}
    for i in range(5):
        try:
            response = requests.post(EMBED_API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()['embeddings']
        except:
            time.sleep(2)
    return [[0] * 4096]

def submit_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=20)
        return response.json().get("score", 0) if response.status_code == 200 else 0
    except:
        return 0

# === 2. 切塊邏輯 (解決 TypeError 與 POINTS 5 問題) ===
def get_chunks(method, content, embeddings_tool):
    if method == "固定大小":
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="")
        return [d.page_content for d in splitter.create_documents([content])]
    
    elif method == "滑動視窗":
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return [d.page_content for d in splitter.create_documents([content])]
    
    elif method == "語義切塊":
        # 💡 解決方案：不使用 sentence_splitter 參數，改為預先手動分句
        # 使用正則表達式按中文標點符號切分
        sentences = re.split(r'(?<=[。！？\n])', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sem_splitter = SemanticChunker(
            embeddings_tool, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50
        )
        
        # 傳入句子列表 (list of strings) 而非單一長字串
        # 這樣 Chunker 會針對這些句子進行語義聚合
        docs = sem_splitter.create_documents(sentences)
        return [d.page_content for d in docs]

# === 3. 主執行流程 ===
def run_evaluation():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    questions_df = pd.read_csv("questions.csv")
    q_ids = questions_df['q_id'].tolist()
    q_texts = questions_df['questions'].tolist()
    
    methods_config = {
        "固定大小": f"{STUDENT_ID}_fixed_size",
        "滑動視窗": f"{STUDENT_ID}_sliding_window",
        "語義切塊": f"{STUDENT_ID}_semantic"
    }
    
    results_for_csv = []
    summary_data = []
    embeddings_tool = CustomEmbeddings()

    print(f"📡 正在獲取 {len(q_texts)} 個問題的向量...")
    all_q_vectors = get_embeddings(q_texts)

    for method_zh, coll_name in methods_config.items():
        print(f"\n🛠️ 處理方法: [{method_zh}]")
        
        if client.collection_exists(coll_name):
            client.delete_collection(coll_name)
            time.sleep(1)
        client.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
        )
        time.sleep(1)

        method_chunks = []
        chunk_source_map = {}
        for file_name in data_files:
            if os.path.exists(file_name):
                with open(file_name, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = get_chunks(method_zh, content, embeddings_tool)
                    for c in chunks:
                        method_chunks.append(c)
                        chunk_source_map[c] = file_name
        
        print(f"   📊 POINTS 數量: {len(method_chunks)}")

        if method_chunks:
            chunk_vectors = get_embeddings(method_chunks)
            points = [
                PointStruct(
                    id=uuid.uuid4().hex, 
                    vector=chunk_vectors[i], 
                    payload={"text": method_chunks[i], "source": chunk_source_map[method_chunks[i]]}
                ) for i in range(len(method_chunks))
            ]
            client.upsert(collection_name=coll_name, points=points)

        method_scores = []
        for i, q_vec in enumerate(all_q_vectors):
            search_res = client.query_points(collection_name=coll_name, query=q_vec, limit=1).points
            if search_res:
                hit = search_res[0]
                retrieved_text = hit.payload['text']
                score = submit_and_get_score(q_ids[i], retrieved_text)
                method_scores.append(score)
                
                results_for_csv.append({
                    "id": uuid.uuid4().hex[:8],
                    "q_id": q_ids[i],
                    "method": method_zh,
                    "retrieve_text": retrieved_text,
                    "score": score,
                    "source": hit.payload['source']
                })
        
        avg = sum(method_scores)/len(method_scores) if method_scores else 0
        summary_data.append({"方法": method_zh, "平均分數": f"{avg:.4f}"})
        print(f"   ✨ 完成！平均分: {avg:.4f}")

    output_name = f"day5/{STUDENT_ID}_RAG_HW_01.csv"
    os.makedirs("day5", exist_ok=True)
    pd.DataFrame(results_for_csv).to_csv(output_name, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全部完成！結果已儲存至: {output_name}")
    print(pd.DataFrame(summary_data))

if __name__ == "__main__":
    run_evaluation()