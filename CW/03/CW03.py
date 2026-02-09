import os
import glob
import pandas as pd
import uuid
import time
import requests
from typing import List

# LangChain 相關組件
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# === 1. 配置與初始化 ===
VLM_BASE_URL = "https://ws-05.huannago.com/v1"
VLM_MODEL = "google/gemma-3-27b-it"
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "gemma_multi_turn_rag_v2" # 建議換個名字避免衝突

llm = ChatOpenAI(
    base_url=VLM_BASE_URL,
    api_key="YOUR_API_KEY", # ⚠️ 執行前請確認 API Key
    model=VLM_MODEL,
    temperature=0,
    timeout=120
)

client = QdrantClient(url="http://localhost:6333")

# === 2. 向量化工具函數 ===
def get_embeddings(texts: List[str]) -> List[List[float]]:
    payload = {"texts": texts, "normalize": True}
    try:
        response = requests.post(EMBED_URL, json=payload, timeout=60)
        return response.json()["embeddings"]
    except Exception as e:
        print(f"❌ Embedding 失敗: {e}")
        return [[0]*4096] * len(texts)

# === 3. 初始化知識庫 (優化切塊與來源標註) ===
def initialize_db():
    print("\n" + "="*50)
    print("📡 [步驟 1/2] 正在初始化本地 Qdrant 知識庫...")
    
    # 獲取維度
    dim = len(get_embeddings(["test"])[0])
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    
    # 抓取目前資料夾下所有 data_0x.txt
    file_paths = glob.glob("data_0*.txt")
    # 優化：Chunk 大小調整為 400，重疊 80 以保留更多上下文
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    all_points = []
    
    for path in file_paths:
        file_name = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = splitter.split_text(content)
            vectors = get_embeddings(chunks)
            for chunk, vec in zip(chunks, vectors):
                all_points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"text": chunk, "source": file_name}
                ))
    
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"✅ 知識庫準備完成，共匯入 {len(all_points)} 個片段。")

# === 4. 執行多輪 RAG 任務 (優化 Prompt) ===
def run_rag_task():
    input_file = "Re_Write_questions.csv"
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    session_history = {} 
    final_answers = []
    final_sources = []

    print("\n🚀 [步驟 2/2] 開始處理問題集...")

    for index, row in df.iterrows():
        cid = str(row['conversation_id'])
        original_q = str(row['questions']) 
        history_list = session_history.get(cid, [])

        # 轉為字串供 Prompt 使用
        history_str = "\n".join([f"問：{h['q']}\n答：{h['a']}" for h in history_list[-2:]]) # 只取最後兩輪避免過長

        # --- 優化後的 Step 1: Query Rewrite ---
        rewrite_prompt = f"""你是一個 RAG 查詢重寫專家。請根據對話歷史，將「最新問題」改寫成一個具備完整主詞、且適合向量搜尋的「繁體中文獨立搜尋句」。
【注意】：
- 指代消解：將「它」、「那邊」、「這個」替換為具體名詞（如 Google N4A, 日本流感）。
- 如果最新問題已經很完整，則微調即可。
- 絕對不要回答問題，只要輸出重寫後的搜尋語句。

[對話歷史]:
{history_str if history_str else "無"}

[最新問題]:
{original_q}

請直接輸出搜尋語句："""

        rewritten_q = llm.invoke(rewrite_prompt).content.strip()

        # --- Step 2: Retrieval ---
        q_vec = get_embeddings([rewritten_q])[0]
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=4
        ).points
        
        context_str = "\n".join([hit.payload['text'] for hit in search_results])
        top_source = search_results[0].payload['source'] if search_results else "未知"

        # --- 優化後的 Step 3: Generation (嚴格控制回答範圍) ---
        final_prompt = f"""你是一位專業助手。請嚴格根據「參考資訊」回答「用戶問題」。
【規則】：
1. 若參考資訊中沒有答案，請直接回答：「抱歉，根據目前的資料庫，我無法回答這個問題。」，絕對不要憑空想像。
2. 回答必須簡潔、準確且專業。
3. 優先回答問題核心。

【參考資訊】：
{context_str}

【用戶問題】：{rewritten_q}

回答："""
        
        answer = llm.invoke(final_prompt).content.strip()
        
        # 更新歷史
        if cid not in session_history: session_history[cid] = []
        session_history[cid].append({"q": original_q, "a": answer})
        
        final_answers.append(answer)
        final_sources.append(top_source)
        
        print(f"Q{index+1} (ID:{cid}): {original_q} -> [重寫]: {rewritten_q}")

    # 儲存結果
    df['answer'] = final_answers
    df['source'] = final_sources
    df.to_csv("Re_Write_questions_result_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 處理完成！結果存於: Re_Write_questions_result_v2.csv")

if __name__ == "__main__":
    initialize_db()
    run_rag_task()