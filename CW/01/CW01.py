import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# --- 1. 初始化連線 ---
print(">>> 正在連接到 Qdrant 伺服器...")
client = QdrantClient(url="http://localhost:6333")

# --- 2. Embedding 函數化 ---
def get_embedding(text_list):
    """
    將文字列表轉換為向量列表
    """
    url = "https://ws-04.wade0426.me/embed"
    data = {
        "texts": text_list,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['embeddings']
    else:
        print(f"!!! API 請求失敗：{response.status_code}")
        return None

# --- 3. 動態計算維度 (不寫死) ---
print(">>> 正在偵測 Embedding 模型維度...")
test_vec = get_embedding(["Dimension Check"])
if test_vec:
    dynamic_size = len(test_vec[0])
    print(f"成功偵測！模型維度為: {dynamic_size}\n")
else:
    raise Exception("無法取得測試向量，請檢查 API 連線。")

# --- 4. 準備 Collection 與資料度量 ---
# 定義三種度量方式
metrics = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclidean": Distance.EUCLID
}

data_source = [
    {"text": "人工智慧很有趣", "category": "AI"},
    {"text": "機器學習是未來趨勢", "category": "AI"},
    {"text": "向量資料庫適合處理非結構化資料", "category": "Database"},
    {"text": "Qdrant 支援高效過濾檢索", "category": "Database"},
    {"text": "Python 是開發 AI 的首選語言", "category": "Programming"}
]

# 批次取得所有資料的向量
all_texts = [item["text"] for item in data_source]
all_embeddings = get_embedding(all_texts)

# --- 5. 批次建立與上傳 ---
for name, dist_type in metrics.items():
    col_name = f"collection_{name.lower()}"
    
    # 建立/重置 Collection
    if client.collection_exists(col_name):
        client.delete_collection(col_name)
    
    print(f">>> 建立 Collection: {col_name} (度量: {name}, 維度: {dynamic_size})")
    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=dynamic_size, distance=dist_type),
    )
    
    # 批次封裝 (Batching)
    points = [
        PointStruct(id=i+1, vector=all_embeddings[i], payload=data_source[i])
        for i in range(len(data_source))
    ]
    
    # 單次批次上傳
    client.upsert(collection_name=col_name, points=points)

# --- 6. 三種度量結果比較展示 ---
query_text = "AI開發，與學習python"
print(f"\n" + "="*60)
print(f"查詢字句：'{query_text}'")
print("="*60)

query_vector = get_embedding([query_text])[0]

for name in metrics.keys():
    col_name = f"collection_{name.lower()}"
    
    # 搜尋：不限單一分類，涵蓋 AI、Database、Programming
    search_result = client.query_points(
        collection_name=col_name,
        query=query_vector,
        query_filter=Filter(
            should=[
                FieldCondition(key="category", match=MatchValue(value="AI")),
                FieldCondition(key="category", match=MatchValue(value="Database")),
                FieldCondition(key="category", match=MatchValue(value="Programming"))
            ]
        ),
        limit=3
    )
    
    print(f"\n【度量模式：{name}】")
    for p in search_result.points:
        print(f" -> 分數: {p.score:8.4f} | 分類: {p.payload['category']:12} | 內容: {p.payload['text']}")

print("\n>>> 所有流程已完成。")