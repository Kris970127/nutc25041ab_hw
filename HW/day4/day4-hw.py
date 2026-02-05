import os
import base64
import operator
import requests
import json
from typing import Annotated, List, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# --- 1. 核心模型初始化 ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="YOUR_API_KEY", # 請更換為您的金鑰
    model="google/gemma-3-27b-it",
    temperature=0
)

# --- 2. 定義狀態 (State) ---
class AgentState(TypedDict):
    input: str
    queries: List[str]
    knowledge_base: Annotated[list, operator.add]
    search_results: List[dict]
    is_sufficient: bool
    round: int
    final_answer: str

# --- 3. 核心工具函數 ---

def search_searxng(query: str):
    """執行搜尋引擎檢索"""
    url = "https://puli-8080.huannago.com/search"
    params = {"q": query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json().get('results', [])[:5] # 多取幾筆供篩選
    except:
        return []

def vlm_read_website(url: str, title: str):
    """使用 Playwright 進行視覺化閱讀"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # 增加等待時間確保內容加載
            page.goto(url, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(3000) 
            screenshot_b64 = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()

            msg = [
                {"type": "text", "text": f"網頁標題：{title}。請摘要這篇報導中關於「發售日期、延期紀錄、官方公告時間」的具體事實。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
            ]
            return llm.invoke([HumanMessage(content=msg)]).content
    except Exception as e:
        return f"視覺讀取失敗: {str(e)}"

# --- 4. LangGraph 節點實作 ---

def check_cache_node(state: AgentState):
    print(f"🔍 [Cache] 檢查快取：{state['input']}")
    return {"round": 0, "knowledge_base": []}

def planner_node(state: AgentState):
    current_round = state.get("round", 0)
    MAX_ROUNDS = 3 
    
    print(f"\n🧠 [Think] Round {current_round}")
    
    if current_round >= MAX_ROUNDS:
        return {"is_sufficient": True}

    if not state.get("knowledge_base"):
        return {"is_sufficient": False, "round": current_round + 1}
    
    context = "\n".join(state["knowledge_base"])
    # 強化判斷邏輯，要求檢查是否有矛盾或不完整
    prompt = f"問題：{state['input']}\n目前查到的資訊：{context}\n這些資訊是否涵蓋了該問題的所有歷史變動或次數？請回答 Y 或 N。"
    res = llm.invoke(prompt)
    
    is_ok = "Y" in res.content.upper()
    print(f"{'✅ 資訊已足夠' if is_ok else '❌ 資訊仍不足，繼續追蹤'}")
    return {"is_sufficient": is_ok, "round": current_round + 1}

def query_gen_node(state: AgentState):
    # 針對延期問題，生成更具追溯性的關鍵字
    prompt = f"針對問題 '{state['input']}'，請生成一個能搜到『歷史變動』或『多次紀錄』的繁體中文搜尋關鍵字（例如：GTA 6 歷次延期 整理）。"
    res = llm.invoke(prompt)
    query = res.content.strip().replace('"', '')
    print(f"🔑 生成關鍵字：{query}")
    return {"queries": [query]}

def search_tool_node(state: AgentState):
    query = state["queries"][-1]
    print(f"🌐 訪問：執行 SearXNG 網路搜尋...")
    return {"search_results": search_searxng(query)}

def vlm_processing_node(state: AgentState):
    """優化：一次讀取前 2 筆結果，確保不漏掉舊資訊"""
    new_info = []
    results = state.get("search_results", [])
    
    # 讀取前 2 筆不同的來源
    for i in range(min(2, len(results))):
        target = results[i]
        print(f"📸 [VLM] 啟動視覺閱讀 ({i+1}/2)：{target.get('title')[:20]}...")
        summary = vlm_read_website(target['url'], target.get('title', '無標題'))
        new_info.append(f"【來源 {i+1}】: {target['url']}\n【摘要】: {summary}\n")
    
    print(f"📝 內容已成功存入知識庫")
    return {"knowledge_base": new_info}

def final_answer_node(state: AgentState):
    print(f"\n🏁 [Output] 正在生成最終查證回答...")
    context = "\n".join(state.get("knowledge_base", []))
    
    prompt = f"""
    請根據以下多個來源的資訊，嚴謹地回答問題：{state['input']}
    
    要求：
    1. 若不同來源提到的次數或日期不同，請完整列出變動歷程。
    2. 使用繁體中文，保留專有名詞。
    3. 採用「條列式」說明各階段的日期。
    4. 若有明確的延期次數，請直接指出。
    
    參考資訊：
    {context}
    """
    res = llm.invoke(prompt)
    return {"final_answer": res.content}

# --- 5. 構建圖表 ---
workflow = StateGraph(AgentState)
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("vlm_processing", vlm_processing_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")
workflow.add_edge("check_cache", "planner")

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
print(app.get_graph().draw_ascii())
if __name__ == "__main__":
    user_q = input("🔍 請輸入您想查證的問題: ")
    if user_q.strip():
        final_state = app.invoke({
            "input": user_q, 
            "knowledge_base": [], 
            "queries": [], 
            "search_results": [],
            "round": 0
        })
        print("\n🎯 【最終查證結果】")
        print(final_state.get("final_answer", "未能生成答案"))