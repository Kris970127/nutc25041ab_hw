import requests
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. 初始化與 API 設定 ---
client = QdrantClient(url="http://localhost:6333")

def get_embedding(text_list):
    url = "https://ws-04.wade0426.me/embed"
    res = requests.post(url, json={"texts": text_list, "normalize": True, "batch_size": 32})
    return res.json()['embeddings'] if res.status_code == 200 else None

# 動態偵測維度
dynamic_size = len(get_embedding(["Dimension Check"])[0])

# --- 2. 定義長文本切塊邏輯 (針對 text.txt) ---
def fixed_size_chunking(text, size=80):
    """固定長度：直接每 size 個字切一段"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def sliding_window_chunking(text, size=80, overlap=35):
    """滑動視窗：移動步長為 (size - overlap)，確保內容重疊"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += (size - overlap)
        if start >= len(text): break
    return chunks

# --- 3. 執行長文本處理與物理內容印出 ---
with open("text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().replace('\n', '')

f_chunks = fixed_size_chunking(raw_text)
s_chunks = sliding_window_chunking(raw_text)

print("\n" + "="*50)
print("【物理切塊內容驗證 - 證明切出來的不一樣】")
print(f"Fixed [1] 開頭: {f_chunks[1][:30]}...")
print(f"Sliding [1] 開頭: {s_chunks[1][:30]}... (這一段應該包含上一塊的結尾文字)")
print("="*50)

# --- 4. 處理表格摘要 (參考 Prompt_table_v1.txt 邏輯) ---
# 這裡根據你的 table_html.html 與 table_txt.md 內容手動模擬摘要輸出
table_summaries = [
    {
        "text": "識別主體：台中科大2026年度發展報告。關鍵數據：就業率達99.2%。重點項目：三民校區(虛實共構商圈)、民生校區(基因編輯護理)、南屯航太園區。趨勢：校區機能重劃以應對少子化。",
        "src": "table_html.html"
    },
    {
        "text": "識別主體：台積電2027年度策略展望。關鍵數據：HPC營收佔比將突破55%。技術趨勢：1.4奈米(A14)於2027年風險試產，推動CoWoS-X封裝與矽光子技術解決傳輸瓶頸。",
        "src": "table_txt.md"
    }
]

# --- 5. 整合資料與批次嵌入 ---
all_payloads = []
for c in f_chunks: all_payloads.append({"text": c, "method": "Fixed", "src": "text.txt"})
for c in s_chunks: all_payloads.append({"text": c, "method": "Sliding", "src": "text.txt"})
for s in table_summaries: all_payloads.append({"text": s["text"], "method": "Summary", "src": s["src"]})

# 一次性取得所有 Embedding
chunk_embeddings = get_embedding([d["text"] for d in all_payloads])

# 建立三種度量衡 Collection
metrics = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclidean": Distance.EUCLID}

for name, dist_type in metrics.items():
    col_name = f"cw02_final_{name.lower()}"
    if client.collection_exists(col_name):
        client.delete_collection(col_name)
    
    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=dynamic_size, distance=dist_type)
    )
    
    # 批次嵌入 (Batching)
    points = [PointStruct(id=i, vector=chunk_embeddings[i], payload=all_payloads[i]) for i in range(len(all_payloads))]
    client.upsert(collection_name=col_name, points=points)
    print(f">>> {col_name} (度量:{name}) 批次嵌入成功。")

# --- 6. 搜尋測試與召回對比 ---
query_text = "台中科大的旗艦計畫內容與就業率數據為何？"
query_v = get_embedding([query_text])[0]

print("\n" + "#"*60)
print(f"綜合查詢測試：{query_text}")
print("#"*60)

for m_name in metrics.keys():
    # 增加 limit 到 4，確保能看到不同的 Method
    res = client.query_points(collection_name=f"cw02_final_{m_name.lower()}", query=query_v, limit=4)
    print(f"\n【度量模式：{m_name}】")
    for p in res.points:
        print(f" -> 分數: {p.score:.4f} | 方法: {p.payload['method']:8} | 來源: {p.payload['src']:15}")
        print(f"    內容: {p.payload['text'][:50]}...")