"""召回率测试：执行50个测试用例，计算检索指标"""
import httpx
import json
import time

API_KEY = "zk_415222bca031"
HEADERS = {"X-API-Key": API_KEY, "Origin": "http://localhost:8501"}
BASE_URL = "http://localhost:8000"

# 50个测试用例
TEST_CASES = TEST_CASES = [

    # ======================
    # 文档1 产品参数手册
    # ======================

    {
        "id": "P01",
        "query": "MP12处理器是什么",
        "expected": ["骁龙8 Gen3"]
    },

    {
        "id": "P02",
        "query": "手机电池多大",
        "expected": ["5000mAh"]
    },

    {
        "id": "P03",
        "query": "Nova MP12屏幕刷新率是多少",
        "expected": ["120Hz"]
    },

    {
        "id": "P04",
        "query": "手机支持什么网络",
        "expected": [
            "5G",
            "WiFi 7"
        ]
    },


    # ======================
    # 文档2 产品功能
    # ======================

    {
        "id": "F01",
        "query": "MP12有哪些AI功能",
        "expected": [
            "AI图片生成",
            "AI文本总结",
            "AI会议记录"
        ]
    },

    {
        "id": "F02",
        "query": "MP12玩游戏性能怎么样",
        "expected": [
            "GPU Turbo",
            "游戏插帧"
        ]
    },


    {
        "id": "F03",
        "query": "手机有哪些拍照功能",
        "expected": [
            "夜景增强",
            "人像虚化"
        ]
    },


    # ======================
    # 文档3 售后服务
    # ======================

    {
        "id": "S01",
        "query": "手机保修多久",
        "expected": [
            "两年官方质保"
        ]
    },

    {
        "id": "S02",
        "query": "电池坏了保修吗",
        "expected": [
            "一年保修"
        ]
    },


    {
        "id": "S03",
        "query": "买7天可以换手机吗",
        "expected": [
            "7天内支持换机"
        ]
    },


    # ======================
    # 文档4 产品价格
    # ======================

    {
        "id": "V01",
        "query": "MP12多少钱",
        "expected": [
            "3999元",
            "4999元"
        ]
    },


    {
        "id": "V02",
        "query": "Pro版本配置是什么",
        "expected": [
            "16GB RAM",
            "512GB"
        ]
    },


    # ======================
    # 文档5 FAQ
    # ======================

    {
        "id": "Q01",
        "query": "MP12适合打游戏吗",
        "expected": [
            "骁龙8 Gen3",
            "GPU Turbo"
        ]
    },


    {
        "id": "Q02",
        "query": "支持无线充电吗",
        "expected": [
            "50W无线快充"
        ]
    },


    {
        "id": "Q03",
        "query": "手机防水吗",
        "expected": [
            "IP68"
        ]
    },


    {
        "id": "Q04",
        "query": "普通用户买哪个版本",
        "expected": [
            "标准版"
        ]
    }

]

def test_retrieval():
    results = []
    print("=== 召回率测试 ===\n")

    for tc in TEST_CASES:
        start = time.time()
        try:
            r = httpx.post(f"{BASE_URL}/chat", json={
                "message": tc["query"],
                "chat_history": [],
                "session_id": f"test_{tc['id']}"
            }, headers=HEADERS, timeout=30.0)
            elapsed = time.time() - start
            data = r.json()
            knowledge = data.get("retrieved_knowledge", "")
        except Exception as e:
            elapsed = time.time() - start
            knowledge = ""
            data = {"error": str(e)}

        hit = any(kw in knowledge for kw in tc["expected"]) if tc["expected"] else True

        results.append({
            "id": tc["id"],
            "query": tc["query"],
            "hit": hit,
            "time_ms": round(elapsed * 1000),
            "knowledge_preview": knowledge[:100] if knowledge else "(empty)"
        })

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {tc['id']} {tc['query']}: {elapsed:.1f}s")

    # 统计
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_time = sum(r["time_ms"] for r in results) / total
    fails = [r for r in results if not r["hit"]]

    print(f"\n{'='*50}")
    print(f"总计: {hits}/{total} 命中")
    print(f"召回率: {hits/total*100:.1f}%")
    print(f"平均响应时间: {avg_time:.0f}ms")

    if fails:
        print(f"\n失败用例 ({len(fails)}):")
        for f in fails:
            print(f"  [{f['id']}] {f['query']}: 未命中")

    # 保存结果
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 test_results.json")

    return results

if __name__ == "__main__":
    test_retrieval()
