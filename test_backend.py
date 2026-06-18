import pytest
import pytest_asyncio
import json
import io
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from backend import (
    app, graph, message_buffer, llm_client,
    input_collector, sentiment_analyzer, query_optimizer,
    knowledge_retriever, response_generator, AgentState,
    _keyword_sentiment_fallback, _keyword_query_fallback,
    chunk_text, parse_txt
)

client = TestClient(app)

mock_llm = AsyncOpenAI(api_key="test-key", base_url="https://api.deepseek.com")


def make_state(session_id: str, message: str) -> AgentState:
    return {
        "session_id": session_id,
        "raw_message": message,
        "merged_message": "",
        "sentiment_score": "normal",
        "sentiment_label": "neutral",
        "sentiment_confidence": 0.5,
        "intent": "other",
        "optimized_query": "",
        "keywords": [],
        "retrieved_knowledge": "",
        "reply": "",
        "status": "normal"
    }


class TestInputCollector:
    def setup_method(self):
        message_buffer.clear()

    def test_single_message_not_merged(self):
        state = make_state("s1", "你好")
        result = input_collector(state)
        assert result["merged_message"] == "你好"

    def test_multiple_messages_merged(self):
        import time
        message_buffer["s2"].append((time.time(), "第一条消息"))
        state = make_state("s2", "第二条消息")
        result = input_collector(state)
        assert "第一条消息" in result["merged_message"]
        assert "第二条消息" in result["merged_message"]

    def test_old_messages_expired(self):
        import time
        old_time = time.time() - 10
        message_buffer["s3"].append((old_time, "过期消息"))
        state = make_state("s3", "新消息")
        result = input_collector(state)
        assert result["merged_message"] == "新消息"
        assert len(message_buffer["s3"]) == 1


class TestKeywordFallbacks:
    def test_sentiment_fallback_angry(self):
        assert _keyword_sentiment_fallback("你这个傻逼") == "angry"
        assert _keyword_sentiment_fallback("我很生气") == "angry"
        assert _keyword_sentiment_fallback("垃圾产品") == "angry"

    def test_sentiment_fallback_normal(self):
        assert _keyword_sentiment_fallback("MP12多少钱？") == "normal"
        assert _keyword_sentiment_fallback("你好") == "normal"

    def test_query_fallback(self):
        keywords = _keyword_query_fallback("MP12多少钱？")
        assert "多少钱" in keywords

    def test_query_fallback_empty(self):
        keywords = _keyword_query_fallback("你好")
        assert len(keywords) == 0


class TestSentimentAnalyzer:
    @pytest.mark.asyncio
    async def test_llm_sentiment_angry(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "label": "angry",
            "confidence": 0.95,
            "reason": "客户使用了侮辱性语言"
        })

        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, return_value=mock_response):
                state = make_state("s", "你这个傻逼")
                state["merged_message"] = "你这个傻逼"
                result = await sentiment_analyzer(state)
                assert result["sentiment_score"] == "angry"
                assert result["sentiment_label"] == "angry"
                assert result["sentiment_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_llm_sentiment_neutral(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "label": "neutral",
            "confidence": 0.85,
            "reason": "客户正常询问产品信息"
        })

        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, return_value=mock_response):
                state = make_state("s", "MP12多少钱？")
                state["merged_message"] = "MP12多少钱？"
                result = await sentiment_analyzer(state)
                assert result["sentiment_score"] == "normal"
                assert result["sentiment_label"] == "neutral"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self):
        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, side_effect=Exception("API Error")):
                state = make_state("s", "你这个傻逼")
                state["merged_message"] = "你这个傻逼"
                result = await sentiment_analyzer(state)
                assert result["sentiment_score"] == "angry"
                assert result["sentiment_label"] == "angry"

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "你这个傻逼")
            state["merged_message"] = "你这个傻逼"
            result = await sentiment_analyzer(state)
            assert result["sentiment_score"] == "angry"


class TestQueryOptimizer:
    @pytest.mark.asyncio
    async def test_llm_query_optimization(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "pricing_query",
            "optimized_query": "MP12智能平板价格售价",
            "keywords": ["价格", "售价"]
        })

        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, return_value=mock_response):
                state = make_state("s", "MP12多少钱？")
                state["merged_message"] = "MP12多少钱？"
                result = await query_optimizer(state)
                assert result["intent"] == "pricing_query"
                assert "价格" in result["keywords"]

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self):
        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, side_effect=Exception("API Error")):
                state = make_state("s", "MP12多少钱？")
                state["merged_message"] = "MP12多少钱？"
                result = await query_optimizer(state)
                assert "多少钱" in result["keywords"]

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "MP12多少钱？")
            state["merged_message"] = "MP12多少钱？"
            result = await query_optimizer(state)
            assert "多少钱" in result["keywords"]


class TestKnowledgeRetriever:
    @pytest.mark.asyncio
    async def test_keyword_match(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "摄像头")
            state["merged_message"] = "摄像头"
            state["optimized_query"] = "摄像头"
            result = await knowledge_retriever(state)
            assert "摄像头" in result["retrieved_knowledge"] or "1300万" in result["retrieved_knowledge"]

    @pytest.mark.asyncio
    async def test_semantic_search(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "这个平板怎么样")
            state["merged_message"] = "这个平板怎么样"
            state["optimized_query"] = "这个平板怎么样"
            result = await knowledge_retriever(state)
            assert "MP12" in result["retrieved_knowledge"]


class TestResponseGenerator:
    @pytest.mark.asyncio
    async def test_llm_response_generation(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "您好！MP12智能平板售价2999元，支持12期免息分期。"

        with patch("backend.llm_client", mock_llm):
            with patch.object(mock_llm.chat.completions, 'create', new_callable=AsyncMock, return_value=mock_response):
                state = make_state("s", "test")
                state["merged_message"] = "MP12多少钱"
                state["retrieved_knowledge"] = "MP12售价2999元"
                state["intent"] = "pricing_query"
                state["sentiment_label"] = "neutral"
                result = await response_generator(state)
                assert "MP12" in result["reply"]
                assert result["status"] == "normal"

    @pytest.mark.asyncio
    async def test_llm_fallback_without_knowledge(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "test")
            state["merged_message"] = "test"
            state["retrieved_knowledge"] = ""
            state["intent"] = "other"
            result = await response_generator(state)
            assert "抱歉" in result["reply"]

    @pytest.mark.asyncio
    async def test_llm_fallback_with_knowledge(self):
        with patch("backend.llm_client", None):
            state = make_state("s", "test")
            state["merged_message"] = "test"
            state["retrieved_knowledge"] = "MP12售价2999元"
            state["intent"] = "pricing_query"
            result = await response_generator(state)
            assert "MP12" in result["reply"]
            assert "2999" in result["reply"]


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_normal_chat_flow(self):
        with patch("backend.llm_client", None):
            result = await graph.ainvoke(make_state("test_graph_1", "MP12多少钱？"))
            assert result["status"] == "normal"
            assert result["sentiment_score"] == "normal"
            assert "MP12" in result["reply"] or "2999" in result["reply"]

    @pytest.mark.asyncio
    async def test_angry_triggers_handover(self):
        with patch("backend.llm_client", None):
            result = await graph.ainvoke(make_state("test_graph_2", "你这个傻逼"))
            assert result["status"] == "human"
            assert result["sentiment_score"] == "angry"
            assert "人工" in result["reply"]

    @pytest.mark.asyncio
    async def test_return_triggers_handover(self):
        with patch("backend.llm_client", None):
            result = await graph.ainvoke(make_state("test_graph_3", "退货"))
            assert result["status"] == "human"
            assert result["sentiment_score"] == "angry"


class TestChatEndpoint:
    def test_normal_chat(self):
        response = client.post(
            "/chat",
            json={"session_id": "test_session", "message": "MP12多少钱？"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert data["status"] == "normal"
        assert data["sentiment_score"] == "normal"
        assert "MP12" in data["reply"] or "2999" in data["reply"]

    def test_insult_detection(self):
        response = client.post(
            "/chat",
            json={"session_id": "test_session2", "message": "你这个傻逼"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "human"
        assert data["sentiment_score"] == "angry"
        assert "人工" in data["reply"]

    def test_response_has_llm_fields(self):
        response = client.post(
            "/chat",
            json={"session_id": "test_session5", "message": "MP12怎么样"}
        )
        data = response.json()
        assert "sentiment_label" in data
        assert "sentiment_confidence" in data
        assert "intent" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestResponseFormat:
    def test_response_has_required_fields(self):
        response = client.post(
            "/chat",
            json={"session_id": "test_format", "message": "你好"}
        )
        data = response.json()
        assert isinstance(data["reply"], str)
        assert data["status"] in ["normal", "human"]
        assert "sentiment_score" in data
        assert "sentiment_label" in data
        assert "intent" in data
        assert "retrieved_knowledge" in data

    def test_invalid_request_returns_json(self):
        response = client.post(
            "/chat",
            json={"invalid_field": "test"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestChunking:
    def test_basic_chunk(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        assert len(chunks[0]) == 500

    def test_overlap(self):
        text = "ABCDEFGHIJ" * 100
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert chunks[0][-50:] in chunks[1]

    def test_short_text_single_chunk(self):
        text = "短文本"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_empty_chunks_stripped(self):
        text = "hello\n\n\n\n\n\nworld"
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert all(c.strip() for c in chunks)


class TestUploadEndpoint:
    def test_upload_txt(self):
        content = "这是测试内容，用于验证文件上传功能。" * 50
        response = client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks"] > 0

    def test_upload_unsupported_format(self):
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(b"data"), "application/pdf")}
        )
        assert response.status_code == 400

    def test_upload_empty_file(self):
        response = client.post(
            "/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        )
        assert response.status_code == 400

    def test_list_uploads(self):
        response = client.get("/upload/list")
        assert response.status_code == 200
        assert "files" in response.json()

    def test_delete_upload(self):
        content = "测试删除功能内容" * 100
        client.post(
            "/upload",
            files={"file": ("del_test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
        )
        response = client.delete("/upload/del_test.txt")
        assert response.status_code == 200
        assert response.json()["deleted_chunks"] > 0


class TestRetrievalWithUploadedKnowledge:
    @pytest.mark.asyncio
    async def test_uploaded_content_retrieved(self):
        content = "MP12智能平板支持卫星通信功能，可在无网络环境下发送紧急求救信号。" * 10
        client.post(
            "/upload",
            files={"file": ("sat.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
        )
        with patch("backend.llm_client", None):
            state = make_state("s", "卫星通信")
            state["merged_message"] = "卫星通信"
            state["optimized_query"] = "卫星通信"
            result = await knowledge_retriever(state)
            assert "卫星" in result["retrieved_knowledge"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
