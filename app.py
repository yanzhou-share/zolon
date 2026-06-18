import json
import os
import streamlit as st
import httpx
import uuid

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
SESSIONS_FILE = os.path.join(SESSIONS_DIR, "sessions.json")

os.makedirs(SESSIONS_DIR, exist_ok=True)


def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def get_session_messages(session_id: str) -> list:
    sessions = load_sessions()
    return sessions.get(session_id, [])


def save_session_messages(session_id: str, messages: list):
    sessions = load_sessions()
    sessions[session_id] = messages
    save_sessions(sessions)


def delete_session(session_id: str):
    sessions = load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        save_sessions(sessions)


st.set_page_config(page_title="AI 智能销售终端", page_icon="🤖", layout="wide")

st.title("AI 智能销售终端")

BACKEND_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = get_session_messages(st.session_state.session_id)

with st.sidebar:
    st.header("会话信息")
    st.write(f"会话ID: {st.session_state.session_id[:8]}...")
    st.write(f"消息数量: {len(st.session_state.messages)}")

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
            if selected_id and selected_id != st.session_state.session_id:
                st.session_state.session_id = selected_id
                st.session_state.messages = get_session_messages(selected_id)
                st.rerun()

    if st.button("清除当前对话"):
        delete_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if st.button("清除所有对话"):
        save_sessions({})
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.header("文档上传")
    uploaded_file = st.file_uploader(
        "上传知识文档 (.txt / .docx)",
        type=["txt", "docx"],
        help="支持 TXT 和 DOCX 格式，最大 10MB"
    )

    if uploaded_file:
        if st.button("上传到知识库"):
            with st.spinner("正在处理文档..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = httpx.post(f"{BACKEND_URL}/upload", files=files, timeout=60.0)
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
        resp = httpx.get(f"{BACKEND_URL}/upload/list", timeout=5.0)
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
                            httpx.delete(f"{BACKEND_URL}/upload/{f['filename']}", timeout=5.0)
                            st.rerun()
    except Exception:
        pass

st.markdown("---")

chat_container = st.container()

with chat_container:
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

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{BACKEND_URL}/chat/stream",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt
                    }
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
