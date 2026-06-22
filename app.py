import json
import os
import streamlit as st
import httpx
import uuid

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_sessions() -> dict:
    sessions = {}
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            filepath = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    sessions[session_id] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return sessions


def get_session_messages(session_id: str) -> list:
    filepath = _session_path(session_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_session_messages(session_id: str, messages: list):
    filepath = _session_path(session_id)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)


def delete_session(session_id: str):
    filepath = _session_path(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)


def save_sessions(sessions: dict):
    for sid, messages in sessions.items():
        save_session_messages(sid, messages)


def delete_all_sessions():
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            os.remove(os.path.join(SESSIONS_DIR, filename))


st.set_page_config(page_title="AI 智能销售终端", page_icon="🤖", layout="wide")

BACKEND_URL = "http://localhost:8000"
DEFAULT_GREETING = "我是您的AI智能助手小龙，随时为您服务！"

MERCHANTS_FILE = os.path.join(os.path.dirname(__file__), "merchants.json")

def load_merchants() -> dict:
    if os.path.exists(MERCHANTS_FILE):
        with open(MERCHANTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"merchants": {}}

def get_merchant_info(api_key: str) -> dict:
    merchants = load_merchants()
    return merchants.get("merchants", {}).get(api_key, {})

def get_greeting(api_key: str) -> str:
    info = get_merchant_info(api_key)
    name = info.get("assistant_name", "小龙")
    return f"我是您的AI智能助手{name}，随时为您服务！"

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "merchant_sessions" not in st.session_state:
    st.session_state["merchant_sessions"] = {}
if "edit_merchant_key" not in st.session_state:
    st.session_state["edit_merchant_key"] = ""

api_key = st.session_state.get("api_key", "")
if api_key:
    if api_key not in st.session_state["merchant_sessions"]:
        st.session_state["merchant_sessions"][api_key] = str(uuid.uuid4())
    st.session_state["session_id"] = st.session_state["merchant_sessions"][api_key]
else:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = get_session_messages(st.session_state.get("session_id", ""))
    if not st.session_state.messages:
        greeting = get_greeting(st.session_state.get("api_key", ""))
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        save_session_messages(st.session_state["session_id"], st.session_state.messages)

with st.sidebar:
    st.header("商户管理")
    merchants_data = load_merchants()
    merchant_list = merchants_data.get("merchants", {})
    if merchant_list:
        merchant_options = [f"{v['name']} ({k})" for k, v in merchant_list.items()]
        current_idx = 0
        current_api_key = st.session_state.get("api_key", "")
        if current_api_key:
            for i, k in enumerate(merchant_list.keys()):
                if k == current_api_key:
                    current_idx = i
                    break
        selected_merchant = st.selectbox("选择商户", options=merchant_options, index=current_idx)
        selected_api_key = selected_merchant.split("(")[-1].rstrip(")")
        if selected_api_key != st.session_state.get("api_key", ""):
            st.session_state["api_key"] = selected_api_key
            if selected_api_key not in st.session_state["merchant_sessions"]:
                st.session_state["merchant_sessions"][selected_api_key] = str(uuid.uuid4())
            st.session_state["session_id"] = st.session_state["merchant_sessions"][selected_api_key]
            st.session_state.messages = get_session_messages(st.session_state["session_id"])
            if not st.session_state.messages:
                greeting = get_greeting(selected_api_key)
                st.session_state.messages = [{"role": "assistant", "content": greeting}]
                save_session_messages(st.session_state["session_id"], st.session_state.messages)
            st.rerun()
        st.caption(f"API Key: {selected_api_key[:12]}...")
        assistant_name = get_merchant_info(selected_api_key).get("assistant_name", "小龙")
        st.caption(f"助手名称: {assistant_name}")
    else:
        st.warning("暂无商户，请先创建")
        st.session_state["api_key"] = ""

    with st.expander("➕ 添加商户"):
        new_name = st.text_input("商户名称", key="new_merchant_name")
        new_assistant = st.text_input("助手名称", value="小龙", key="new_assistant_name")
        new_origins = st.text_input("允许域名 (逗号分隔)", key="new_merchant_origins", placeholder="https://example.com")
        new_rate = st.number_input("每分钟请求上限", value=100, min_value=1, key="new_rate_limit")
        if st.button("创建商户", key="btn_create_merchant"):
            if new_name:
                api_key = "zk_" + __import__("secrets").token_hex(6)
                origins = ["http://localhost:8501"]
                if new_origins:
                    origins.extend([o.strip() for o in new_origins.split(",") if o.strip()])
                merchants_data["merchants"][api_key] = {
                    "name": new_name,
                    "assistant_name": new_assistant or "小龙",
                    "status": "active",
                    "created_at": __import__("datetime").datetime.now().isoformat(),
                    "allowed_origins": origins,
                    "rate_limit": new_rate,
                    "daily_limit": 5000
                }
                with open(MERCHANTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(merchants_data, f, ensure_ascii=False, indent=2)
                st.success(f"创建成功！API Key: {api_key}")
                st.rerun()
            else:
                st.error("请输入商户名称")

    if merchant_list:
        with st.expander("✏️ 编辑商户"):
            edit_key = st.selectbox("选择要编辑的商户", options=list(merchant_list.keys()), format_func=lambda k: f"{merchant_list[k]['name']} ({k})", key="edit_merchant_select")
            if edit_key:
                info = merchant_list[edit_key]
                edit_name = st.text_input("商户名称", value=info.get("name", ""), key=f"edit_name_{edit_key}")
                edit_assistant = st.text_input("助手名称", value=info.get("assistant_name", "小龙"), key=f"edit_assistant_{edit_key}")
                edit_status = st.selectbox("状态", options=["active", "inactive"], index=0 if info.get("status") == "active" else 1, key=f"edit_status_{edit_key}")
                edit_origins = st.text_input("允许域名 (逗号分隔)", value=", ".join(info.get("allowed_origins", [])), key=f"edit_origins_{edit_key}")
                edit_rate = st.number_input("每分钟请求上限", value=info.get("rate_limit", 100), min_value=1, key=f"edit_rate_{edit_key}")
                edit_daily = st.number_input("每日请求上限", value=info.get("daily_limit", 5000), min_value=1, key=f"edit_daily_{edit_key}")
                if st.button("保存修改", key="btn_save_merchant"):
                    merchants_data["merchants"][edit_key]["name"] = edit_name
                    merchants_data["merchants"][edit_key]["assistant_name"] = edit_assistant
                    merchants_data["merchants"][edit_key]["status"] = edit_status
                    merchants_data["merchants"][edit_key]["allowed_origins"] = [o.strip() for o in edit_origins.split(",") if o.strip()]
                    merchants_data["merchants"][edit_key]["rate_limit"] = edit_rate
                    merchants_data["merchants"][edit_key]["daily_limit"] = edit_daily
                    with open(MERCHANTS_FILE, "w", encoding="utf-8") as f:
                        json.dump(merchants_data, f, ensure_ascii=False, indent=2)
                    st.success("修改已保存")
                    st.rerun()

        with st.expander("🗑️ 删除商户"):
            del_key = st.selectbox("选择要删除的商户", options=list(merchant_list.keys()), format_func=lambda k: f"{merchant_list[k]['name']} ({k})", key="del_merchant_select")
            if del_key and st.button("确认删除", key="btn_delete_merchant", type="secondary"):
                del merchants_data["merchants"][del_key]
                with open(MERCHANTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(merchants_data, f, ensure_ascii=False, indent=2)
                st.success("商户已删除")
                st.rerun()

    st.markdown("---")
    st.header("会话信息")
    st.write(f"会话ID: {st.session_state.get('session_id', '')[:8]}...")
    st.write(f"消息数量: {len(st.session_state.messages)}")

    if st.button("新建对话"):
        st.session_state["session_id"] = str(uuid.uuid4())
        api_key = st.session_state.get("api_key", "")
        if api_key:
            st.session_state["merchant_sessions"][api_key] = st.session_state["session_id"]
        greeting = get_greeting(api_key)
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        save_session_messages(st.session_state["session_id"], st.session_state.messages)
        st.rerun()

    sessions = load_sessions()
    session_list = list(sessions.keys())
    if session_list:
        display_options = ["当前会话"] + [s[:8] + "..." for s in session_list]
        selected = st.selectbox(
            "选择历史会话",
            options=display_options,
            index=0
        )
        if selected != "当前会话":
            idx = display_options.index(selected) - 1
            selected_id = session_list[idx]
            if selected_id and selected_id != st.session_state.get("session_id"):
                st.session_state["session_id"] = selected_id
                st.session_state.messages = get_session_messages(selected_id)
                st.rerun()

    if st.button("清除当前对话"):
        delete_session(st.session_state.get("session_id", ""))
        greeting = get_greeting(st.session_state.get("api_key", ""))
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state["session_id"] = str(uuid.uuid4())
        api_key = st.session_state.get("api_key", "")
        if api_key:
            st.session_state["merchant_sessions"][api_key] = st.session_state["session_id"]
        save_session_messages(st.session_state["session_id"], st.session_state.messages)
        st.rerun()

    if st.button("清除所有对话"):
        delete_all_sessions()
        st.session_state["merchant_sessions"] = {}
        greeting = get_greeting(st.session_state.get("api_key", ""))
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state["session_id"] = str(uuid.uuid4())
        api_key = st.session_state.get("api_key", "")
        if api_key:
            st.session_state["merchant_sessions"][api_key] = st.session_state["session_id"]
        save_session_messages(st.session_state["session_id"], st.session_state.messages)
        st.rerun()

    st.markdown("---")
    st.header("文档上传")
    uploaded_file = st.file_uploader(
        "上传知识文档 (.txt / .docx / .md / .pdf)",
        type=["txt", "docx", "md", "pdf"],
        help="支持 TXT 和 DOCX 格式，最大 10MB"
    )

    if uploaded_file:
        if st.button("上传到知识库"):
            api_key = st.session_state.get("api_key", "")
            if not api_key:
                st.error("请先选择商户")
            else:
                with st.spinner("正在处理文档..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        resp = httpx.post(f"{BACKEND_URL}/upload", files=files, headers={"X-API-Key": api_key, "Origin": "http://localhost:8501"}, timeout=60.0)
                        if resp.status_code == 200:
                            result = resp.json()
                            st.success(f"上传成功！文件被分为 {result['chunks']} 个片段")
                        elif resp.status_code == 413:
                            st.error("文件超过 10MB 限制")
                        elif resp.status_code == 400:
                            st.error(resp.json().get("error", "上传失败"))
                        else:
                            st.error(f"上传失败: {resp.status_code}")
                    except httpx.ConnectError:
                        st.error("无法连接后端服务")
                    except Exception as e:
                        st.error(f"上传异常: {str(e)}")

    try:
        api_key = st.session_state.get("api_key", "")
        if api_key:
            resp = httpx.get(f"{BACKEND_URL}/upload/list", headers={"X-API-Key": api_key, "Origin": "http://localhost:8501"}, timeout=5.0)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                if files:
                    st.subheader("已上传文档")
                    for f in files:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"📄 {f['filename']} ({f['chunks']} 片段)")
                        with col2:
                            if st.button("删除", key=f"del_{f['filename']}"):
                                httpx.delete(f"{BACKEND_URL}/upload/{f['filename']}", headers={"X-API-Key": api_key, "Origin": "http://localhost:8501"}, timeout=5.0)
                                st.rerun()
    except Exception:
        pass

st.markdown("---")

tab_chat, tab_dashboard = st.tabs(["💬 对话", "📊 仪表盘"])

with tab_dashboard:
    st.header("运营仪表盘")
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin123")
    try:
        resp = httpx.get(f"{BACKEND_URL}/admin/dashboard", headers={"X-Admin-Key": ADMIN_SECRET}, timeout=10.0)
        if resp.status_code == 200:
            stats = resp.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总请求量", f"{stats['total_requests']:,}")
            col2.metric("总 Token", f"{stats['total_tokens']:,}")
            col3.metric("活跃商户", f"{stats['active_merchants']}/{stats.get('total_merchants', 0)}")
            col4.metric("平均延迟", f"{stats['avg_latency_ms']:.0f}ms")

            if stats.get("daily_stats"):
                st.subheader("每日趋势")
                import pandas as pd
                df = pd.DataFrame(stats["daily_stats"])
                st.line_chart(df.set_index("date")[["requests", "tokens"]])

            if stats.get("merchant_ranking"):
                st.subheader("商户排行")
                st.dataframe(pd.DataFrame(stats["merchant_ranking"]))
        else:
            st.info("暂无数据")
    except Exception as e:
        st.info(f"仪表盘加载失败: {e}")

    api_key = st.session_state.get("api_key", "")
    if api_key:
        st.subheader("当前商户统计")
        try:
            resp = httpx.get(f"{BACKEND_URL}/admin/dashboard/merchant/{api_key}", headers={"X-Admin-Key": ADMIN_SECRET}, timeout=10.0)
            if resp.status_code == 200:
                m_stats = resp.json()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("请求数", f"{m_stats['total_requests']:,}")
                mc2.metric("Token 用量", f"{m_stats['total_tokens']:,}")
                mc3.metric("平均延迟", f"{m_stats['avg_latency_ms']:.0f}ms")

                resp_bill = httpx.get(f"{BACKEND_URL}/admin/billing/{api_key}", headers={"X-Admin-Key": ADMIN_SECRET}, timeout=10.0)
                if resp_bill.status_code == 200:
                    bill = resp_bill.json()
                    st.caption(f"本月费用: ¥{bill['amount']} (免费额度: {bill['free_quota']:,} tokens)")
        except Exception:
            pass

with tab_chat:
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if "sentiment_label" in msg:
                    sentiment = msg["sentiment_label"]
                    color_map = {"positive": "🟢", "neutral": "🟡", "angry": "🔴"}
                    emoji = color_map.get(sentiment, "⚪")
                    st.caption(f"{emoji} 情绪: {sentiment} | 意图: {msg.get('intent', 'unknown')}")
                if "retrieved_knowledge" in msg and msg["retrieved_knowledge"]:
                    with st.expander("查看检索到的知识"):
                        st.write(msg["retrieved_knowledge"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_session_messages(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        spinner_placeholder = st.empty()
        spinner_placeholder.info("正在思考中...")
        full_response = ""
        sentiment_label = "neutral"
        sentiment_confidence = 0.5
        intent = "other"
        retrieved_knowledge = ""
        status = "normal"

        chat_history = [m for m in st.session_state.messages[:-1]]

        try:
            api_key = st.session_state.get("api_key", "")
            headers = {"X-API-Key": api_key, "Origin": "http://localhost:8501"} if api_key else {}
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{BACKEND_URL}/chat/stream",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "chat_history": chat_history
                    },
                    headers=headers
                ) as response:
                    current_event = ""
                    current_data_lines = []
                    for line in response.iter_lines():
                        if not line:
                            if current_event and current_data_lines:
                                data_payload = "\n".join(current_data_lines)
                                if current_event == "token":
                                    spinner_placeholder.empty()
                                    full_response += data_payload
                                    placeholder.markdown(full_response + "▌")
                                elif current_event == "done":
                                    try:
                                        final = json.loads(data_payload)
                                        full_response = final.get("reply", full_response)
                                        sentiment_label = final.get("sentiment_label", "neutral")
                                        sentiment_confidence = final.get("sentiment_confidence", 0.5)
                                        intent = final.get("intent", "other")
                                        retrieved_knowledge = final.get("retrieved_knowledge", "")
                                        status = final.get("status", "normal")
                                    except json.JSONDecodeError:
                                        pass
                            current_event = ""
                            current_data_lines = []
                            continue
                        if line.startswith("event: "):
                            current_event = line[7:].strip()
                        elif line.startswith("data: "):
                            current_data_lines.append(line[6:])

            placeholder.markdown(full_response)

            if status == "human":
                st.warning("已转接人工客服")

            color_map = {"positive": "🟢", "neutral": "🟡", "angry": "🔴"}
            emoji = color_map.get(sentiment_label, "⚪")
            st.caption(f"{emoji} 情绪: {sentiment_label} | 意图: {intent}")

            if retrieved_knowledge:
                with st.expander("查看检索到的知识"):
                    st.write(retrieved_knowledge)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sentiment_label": sentiment_label,
                "sentiment_confidence": sentiment_confidence,
                "intent": intent,
                "retrieved_knowledge": retrieved_knowledge
            })
            save_session_messages(st.session_state.session_id, st.session_state.messages)

        except httpx.ConnectError:
            error_msg = "无法连接到后端服务，请确保后端已启动 (python backend.py)"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_session_messages(st.session_state.session_id, st.session_state.messages)
        except httpx.ReadTimeout:
            error_msg = "请求超时，请稍后重试"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_session_messages(st.session_state.session_id, st.session_state.messages)
        except Exception as e:
            error_msg = f"请求失败: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_session_messages(st.session_state.session_id, st.session_state.messages)

st.markdown("---")
st.caption("AI 智能销售终端 基于DeepSeek LLM")
