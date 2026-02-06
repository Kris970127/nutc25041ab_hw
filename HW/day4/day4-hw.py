import os
import base64
import operator
import requests
import json
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# --- 1. 核心模型初始化 ---
# 請確保 base_url 與 api_key 正確無誤
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1", 
    api_key="YOUR_API_KEY", 
    model="google/gemma-3-27b-it",
    temperature=0
)

# --- 2. 定義狀態 ---
class AgentState(TypedDict):
    input: str
    queries: List[str]
    knowledge_base: Annotated[list, operator.add]
    search_results: List[dict]
    is_sufficient: bool
    round: int
    missing_info: str
    final_answer: str

# --- 3. 核心工具函數 ---

def search_searxng(query: str):
    """執行搜尋引擎檢索，並預先清理關鍵字"""
    url = "https://puli-8080.huannago.com/search"
    clean_query = query.strip().split('\n')[0].replace('*', '').replace('"', '')
    params = {"q": clean_query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(url, params=params, timeout=15)
        return response.json().get('results', [])[:5]
    except Exception as e:
        print(f"🌐 搜尋引擎連接失敗: {e}")
        return []

def vlm_read_website(url: str, title: str, original_q: str):
    """強化版視覺網頁讀取：模擬真實瀏覽器行為，解決截圖空白問題"""
    try:
        with sync_playwright() as p:
            # 模擬真實瀏覽器環境，避開部分防爬蟲機制
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 800}
            )
            page = context.new_page()
            
            # 延長超時時間並等待 DOM 加載
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(3000) # 給予額外渲染時間
            
            # 自動向下滾動觸發懶加載 (Lazy Loading)
            page.mouse.wheel(0, 800)
            page.wait_for_timeout(1000)

            screenshot_b64 = base64.b64encode(page.screenshot(full_page=False)).decode('utf-8')
            browser.close()

            # 指引 VLM 進行嚴謹的事實提取
            msg = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"網頁標題：{title}\n用戶問題：{original_q}\n請依據『調查員原則』提取證據：\n1. 找出所有具體日期與版本數據。\n2. 識別官方公告與傳聞的區別。\n3. 若提到『延期』，請找原始日期與新日期。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                ]}
            ]
            return llm.invoke(msg).content
    except Exception as e:
        return f"視覺讀取失敗 (來源: {url}): {str(e)}"

# --- 4. 嚴謹節點實作 ---

def planner_node(state: AgentState):
    """決策節點：判斷資訊是否構成完整的證據鏈"""
    current_round = state.get("round", 0)
    MAX_ROUNDS = 3
    print(f"\n🧠 [思考] 第 {current_round} 輪調查")
    
    if current_round >= MAX_ROUNDS: return {"is_sufficient": True}
    if not state.get("knowledge_base"): return {"is_sufficient": False, "round": current_round + 1}
    
    context = "\n".join(state["knowledge_base"])
    prompt = f"""使用者問題：{state['input']}
    現有資料內容：{context}
    
    請以『懷疑論』立場評估：
    1. 是否已有明確的官方數據或日期？
    2. 是否能排除媒體猜測並形成完整時間軸？
    如果已足以結案，請回覆 'DONE'。
    否則，請簡短描述『還缺少的特定拼圖』。"""
    
    res = llm.invoke(prompt).content
    if "DONE" in res.upper():
        return {"is_sufficient": True}
    else:
        print(f"❌ 證據鏈不足：{res[:60]}...")
        return {"is_sufficient": False, "round": current_round + 1, "missing_info": res}

def query_gen_node(state: AgentState):
    """
    究極嚴謹版關鍵字生成
    導入：多方求證、結構化思考、懷疑論、時效性
    """
    history = ", ".join(state.get("queries", []))
    missing = state.get("missing_info", "基礎背景事實")
    
    # 強化的系統提示詞，模仿截圖中的偵探人格
    system_prompt = f"""你是一名頂尖的資深調查員，當前日期是 {datetime.now().strftime('%Y-%m-%d')}。
    你必須遵循以下核心準則來生成搜尋詞：
    - **多方求證**：針對現有說法尋找反向證據或官方來源。
    - **結構化思考**：從歷史變動、財報數據、官方社群等多維度切入。
    - **時效性**：確保搜尋詞能涵蓋最新的動態與歷史的節點。
    
    任務：針對問題『{state['input']}』，補足缺失資訊：『{missing}』。
    要求：僅輸出一個精確的搜尋關鍵字，禁止 Markdown、引號或任何解釋。"""

    res = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"已嘗試過的關鍵字：[{history}]。請給出下一個搜尋方向。")
    ]).content

    query = res.strip().split('\n')[0].replace('*', '').replace('"', '').replace('搜尋關鍵字：', '')
    print(f"🔑 [調查級搜尋]：{query}")
    return {"queries": [query]}

def search_tool_node(state: AgentState):
    return {"search_results": search_searxng(state["queries"][-1])}

def vlm_processing_node(state: AgentState):
    new_info = []
    results = state.get("search_results", [])
    if not results:
        return {"knowledge_base": ["(此輪搜尋未獲取有效網頁)"]}

    for i in range(min(2, len(results))):
        target = results[i]
        print(f"📸 [視覺查證] 正在讀取：{target.get('title')[:20]}...")
        summary = vlm_read_website(target['url'], target.get('title', '無標題'), state['input'])
        new_info.append(f"【來源】: {target['url']}\n【事實摘要】: {summary}\n")
    return {"knowledge_base": new_info}

def final_answer_node(state: AgentState):
    """最終彙整：執行邏輯推理與時間軸排序"""
    print(f"\n🏁 [Final Report] 正在產出嚴謹報告...")
    context = "\n".join(state.get("knowledge_base", []))
    
    prompt = f"""
    你是專業調查分析師。請根據以下蒐集到的零散資訊，為用戶問題『{state['input']}』產出報告。
    
    【推論要求】
    1. 務必建立事件的時間軸 (Timeline)。
    2. 計算發生的次數，並指出每次變動的『前、後』狀態。
    3. 區分官方正式公告 (Official) 與媒體傳聞 (Rumor)。
    
    查證資料內容：
    {context}
    """
    res = llm.invoke(prompt).content
    return {"final_answer": res}

# --- 5. 構建圖表 ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("vlm_processing", vlm_processing_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", 
    lambda x: "end" if x["is_sufficient"] else "search", 
    {"end": "final_answer", "search": "query_gen"}
)
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "vlm_processing")
workflow.add_edge("vlm_processing", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

# --- 6. 互動執行 ---
if __name__ == "__main__":
    print("🕵️ 調查級 Agent 已準備就緒。輸入 'q' 結束對話。")
    while True:
        user_q = input("\n🔍 請輸入您的問題: ")
        if user_q.lower() == 'q': break
        
        try:
            final_state = app.invoke({
                "input": user_q, 
                "knowledge_base": [], 
                "queries": [], 
                "round": 0,
                "missing_info": "",
                "final_answer": ""
            })
            
            print("\n" + "—"*50)
            print(f"🎯 【最終調查報告】\n\n{final_state.get('final_answer')}")
            print("—"*50)
        except Exception as e:
            print(f"🔥 系統執行中斷: {e}")