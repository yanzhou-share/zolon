"""准备测试环境：上传5套测试文档"""
import httpx
import os

API_KEY = "zk_test12345678"
HEADERS = {"X-API-Key": API_KEY, "Origin": "http://localhost:8501"}
BASE_URL = "http://localhost:8000"
DOCS_DIR = "test_docs"

def setup():
    print("=== 上传测试文档 ===")
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "rb") as f:
                files = {"file": (filename, f, "text/plain")}
                r = httpx.post(f"{BASE_URL}/upload", files=files, headers=HEADERS, timeout=60.0)
                result = r.json()
                chunks = result.get("chunks", 0)
                version = result.get("version", "?")
                print(f"  {filename}: v{version}, {chunks} chunks")

    # 验证
    r = httpx.get(f"{BASE_URL}/upload/list", headers=HEADERS, timeout=10.0)
    files = r.json().get("files", [])
    print(f"\n已上传 {len(files)} 个文件:")
    for f in files:
        print(f"  {f['filename']}: {f['chunks']} chunks, v{f['latest_version']}")

if __name__ == "__main__":
    setup()
