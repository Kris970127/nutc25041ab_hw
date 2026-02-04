import time
import requests
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI

# 1. API 基礎配置
ASR_BASE = "https://3090api.huannago.com"
LLM_BASE = "https://ws-02.wade0426.me/v1" #
MODEL_NAME = "google/gemma-3-27b-it"       #
AUTH = ("nutc2504", "nutc2504")

client = OpenAI(api_key="YOUR_API_KEY", base_url=LLM_BASE)

class AgentState(TypedDict):
    wav_path: str
    raw_txt: str
    raw_srt: str
    transcript: str
    summary: str
    final_output: str

# 2. 定義功能節點 (Nodes)
def asr_node(state: AgentState):
    """執行 ASR 轉錄 (整合 Requests 腳本)"""
    print("--- [Node] 執行 ASR 語音辨識 ---")
    create_url = f"{ASR_BASE}/api/v1/subtitle/tasks"
    with open(state["wav_path"], "rb") as f:
        r = requests.post(create_url, files={"audio": f}, timeout=60, auth=AUTH)
    r.raise_for_status()
    task_id = r.json()["id"]
    
    def wait_download(url: str):
        for _ in range(600):
            resp = requests.get(url, timeout=(5, 60), auth=AUTH)
            if resp.status_code == 200: return resp.text
            time.sleep(2)
        return ""
    
    # 同時獲取 TXT 與 SRT 分別給摘要與逐字稿使用
    return {
        "raw_txt": wait_download(f"{ASR_BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"),
        "raw_srt": wait_download(f"{ASR_BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT")
    }

def summarizer_node(state: AgentState):
    """生成重點摘要 (嚴格遵守截圖左側格式)"""
    print("--- [Node] 提取重點摘要 (Executive Summary) ---")
    
    # 這裡將 Prompt 修改為與截圖完全一致的文字排版
    prompt = """
    請根據提供的內容，『嚴格』依照以下 Markdown 格式輸出，且『禁止』包含任何 ```markdown 等標籤：

    # 📓 智慧會議紀錄報告
    ## 🎯 重點摘要 (Executive Summary)
    ## 天下文化 Podcast 摘要 - 《努力但不費力》

    (這裡填入本次會議重點探討內容...)

    **決策結果：** ** (這裡填入決策內容)
    **待辦事項 (Action Items)：**
    * **(標題)** : (內容)
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"{prompt}\n\n原始文本：\n{state['raw_txt']}"}],
        temperature=0
    )
    return {"summary": response.choices[0].message.content.strip()}

def minutes_taker_node(state: AgentState):
    """整理詳細逐字稿 (嚴格遵守截圖右側表格格式)"""
    print("--- [Node] 整理詳細逐字稿 (Table Format) ---")
    
    # 強制要求時間軸格式為 00:00:00 - 00:00:00 並轉為表格
    prompt = """
    請將內容轉為以下表格格式，『禁止』包含任何代碼塊圍欄，時間請改為 '00:00:00 - 00:00:00'：

    ## 📝 詳細記錄 (Detailed Minutes)
    ## 會議發言紀錄 - 天下文化 Podcast

    | **時間** | **發言內容** |
    | :--- | :--- |
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"{prompt}\n\n原始SRT：\n{state['raw_srt']}"}],
        temperature=0
    )
    return {"transcript": response.choices[0].message.content.strip()}

def writer_node(state: AgentState):
    """最終彙整並合併"""
    print("--- [Node] 執行最終彙整 (Writer) ---")
    # 合併兩者，中間使用標準分隔線
    report = f"{state['summary']}\n\n---\n\n{state['transcript']}"
    return {"final_output": report}

# 3. 構建圖結構 (依照課後練習圖構)
workflow = StateGraph(AgentState)
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("asr")

# 並行處理
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# 匯合至 writer
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

workflow.add_edge("writer", END)

app = workflow.compile()

# 4. 執行與輸出
if __name__ == "__main__":
    # 使用你的特定檔案路徑
    config = {"wav_path": "/home/pc-49/Downloads/Podcast_EP14_30s.wav"}
    result = app.invoke(config)
    
    # 輸出成 Markdown 檔案
    output_path = Path("Meeting_Analysis_Report.md")
    output_path.write_text(result["final_output"], encoding="utf-8")
    
    print(f"\n✅ 處理完成！結果已儲存至：{output_path}")