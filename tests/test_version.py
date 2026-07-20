import httpx
import chromadb

API_KEY = "zk_test12345678"
HEADERS = {"X-API-Key": API_KEY, "Origin": "http://localhost:8501"}

# 1. 清空知识库
client = chromadb.PersistentClient(path="D:/xiaomi-mimo-test/zolon/chroma_db")
try:
    client.delete_collection(f"knowledge_{API_KEY}")
    print("1. Deleted old collection")
except:
    print("1. No old collection")

# 2. 上传文件 v1
files = {"file": ("FAQ.txt", b"This is version 1 content about cameras.", "text/plain")}
r = httpx.post("http://localhost:8000/upload", files=files, headers=HEADERS, timeout=60.0)
print(f"2. Upload v1: {r.json()}")

# 3. 上传同名文件 v2
files = {"file": ("FAQ.txt", b"This is version 2 content about cameras and GPS.", "text/plain")}
r = httpx.post("http://localhost:8000/upload", files=files, headers=HEADERS, timeout=60.0)
print(f"3. Upload v2: {r.json()}")

# 4. 查看版本历史
r = httpx.get("http://localhost:8000/upload/FAQ.txt/versions", headers=HEADERS, timeout=10.0)
print(f"4. Versions: {r.json()}")

# 5. 查看文件列表
r = httpx.get("http://localhost:8000/upload/list", headers=HEADERS, timeout=10.0)
print(f"5. File list: {r.json()}")

# 6. 测试检索（应该只返回 v2 内容）
r = httpx.post("http://localhost:8000/chat", json={"message": "GPS", "chat_history": [], "session_id": "test_ver"}, headers=HEADERS, timeout=30.0)
d = r.json()
print(f"6. Chat reply: {d['reply'][:100]}")
print(f"   Retrieved: {d.get('retrieved_knowledge', '')[:100] or '(empty)'}")

# 7. 回滚到 v1
r = httpx.post("http://localhost:8000/upload/FAQ.txt/rollback/1", headers=HEADERS, timeout=10.0)
print(f"7. Rollback: {r.json()}")

# 8. 验证回滚后检索（应该返回 v1 内容）
r = httpx.post("http://localhost:8000/chat", json={"message": "cameras", "chat_history": [], "session_id": "test_ver2"}, headers=HEADERS, timeout=30.0)
d = r.json()
print(f"8. After rollback: {d['reply'][:100]}")
print(f"   Retrieved: {d.get('retrieved_knowledge', '')[:100] or '(empty)'}")
