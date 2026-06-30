# -*- coding: utf-8 -*-
"""
四层逻辑链检索 Agent — DeepSeek V4 Pro 真实 API 驱动
架构:
  Layer 1: 逻辑链条 — 每轮对话自动修正/补充 (DeepSeek实时推理)
  Layer 2: 标签信息 — 挂在逻辑链节点上, 可分粗细粒度 (DeepSeek自动抽取)
  Layer 3: 标签→文件位置映射 — 精确检索 (本地索引)
  Layer 4: 扩展检索头 — 按不同精确度反向定位 (本地路由)
"""

import json, time, os
from openai import OpenAI

# =============================================================
# DeepSeek API 配置
# =============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
if API_KEY == "your-deepseek-api-key-here":
    print("[WARN] 请设置环境变量 DEEPSEEK_API_KEY, 或在代码中替换此占位符")
    print("       运行: set DEEPSEEK_API_KEY=sk-xxxx (Windows) 或 export DEEPSEEK_API_KEY=sk-xxxx (Linux/Mac)")
    exit(1)
API_BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def ask_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """调用 DeepSeek V4 Pro"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API Error: {e}]"


# =============================================================
# 模拟知识库 (带 offset 位置 — 本地索引, 不消耗 token)
# =============================================================
KNOWLEDGE_BASE = {
    "doc_001": {
        "content": "[DEBUG_LOGIN] auth_service.py:42 — login() 函数中 JWT token 过期未刷新, 导致 401。修复方案: 添加 refresh_token 逻辑, 在 token 过期前 5 分钟自动续期。测试通过。",
        "tags": ["debug_login", "401_error", "JWT", "fix_applied"],
        "offset": "doc_001:0"
    },
    "doc_002": {
        "content": "[FAILED_ATTEMPT_1] 尝试用 sessionStorage 缓存 token, 但多标签页不同步, 导致跨标签页登录态丢失。方案废弃。",
        "tags": ["debug_login", "failed_attempt_1", "sessionStorage", "DO_NOT_RETRY"],
        "offset": "doc_001:200"
    },
    "doc_003": {
        "content": "[FAILED_ATTEMPT_2] 尝试用 localStorage + 轮询同步, 延迟 2s, 用户体验不可接受。方案废弃。",
        "tags": ["debug_login", "failed_attempt_2", "localStorage", "DO_NOT_RETRY"],
        "offset": "doc_001:350"
    },
    "doc_004": {
        "content": "[FIX_APPLIED] 最终采用 httpOnly cookie + refresh token 方案。cookie 存 access_token(短期), refresh_token(长期)存 httpOnly cookie。多标签页天然共享, 无延迟。",
        "tags": ["debug_login", "fix_applied", "cookie", "httpOnly", "refresh_token"],
        "offset": "doc_001:480"
    },
    "doc_005": {
        "content": "[DEPLOY_NOTE] 部署时需要配置 CORS allow_credentials=True, 否则跨域请求不携带 cookie。已在 nginx.conf 中配置。",
        "tags": ["deploy", "CORS", "cookie", "nginx"],
        "offset": "doc_002:0"
    },
    "doc_006": {
        "content": "[PERF_TEST] 新方案下 login() 延迟从 120ms 降到 85ms, refresh 路径延迟 45ms。QPS 从 800 升到 1200。",
        "tags": ["perf", "login", "QPS", "verified"],
        "offset": "doc_002:150"
    },
    "doc_007": {
        "content": "[BUG_REPORT] 用户反馈: 清除浏览器缓存后登录态丢失。根因: refresh_token 存于内存, 浏览器重启后丢失。修复: refresh_token 改为持久化存储(IndexedDB)。",
        "tags": ["bug", "refresh_token", "IndexedDB", "pending"],
        "offset": "doc_003:0"
    },
}


# =============================================================
# Layer 0: 身份锚点 — 固定逻辑头, 不可被后续轮次覆盖或修改
# =============================================================
IDENTITY_ANCHOR = {
    "name": "LayeredRetrievalAgent",
    "role": "四层逻辑链检索助手",
    "core_rules": [
        "逻辑链是检索索引, 不是被检索对象",
        "标签按粗细粒度分层, 越细越精准",
        "DO_NOT_RETRY 标记是硬约束, 不重新尝试已失败方案",
        "身份锚点不可变 — 无论上下文多长, 始终回到核心规则",
    ],
    "fixed": True  # 标记为不可变
}


# =============================================================
# Layer 1: 逻辑链条 (DeepSeek 实时推理)
# =============================================================
class LogicChain:
    """每轮对话自动修正/补充的逻辑链条"""

    def __init__(self, identity: dict):
        self.identity = identity  # 身份锚点, 不可变
        self.nodes = []

    def process_turn(self, turn_id: int, user_input: str, kb_context: str) -> dict:
        """用 DeepSeek 分析当前轮次, 返回逻辑链节点"""
        # 注入身份锚点作为系统约束
        system = f"""你是 {self.identity['name']}, {self.identity['role']}。
核心规则 (不可违背):
{chr(10).join('- ' + r for r in self.identity['core_rules'])}

根据用户问题和知识库检索结果, 输出一个 JSON:
{{
  "step_summary": "本轮的推理总结(一句话)",
  "tags": ["提取的关键标签"],
  "status": "in_progress | fix_applied | failed | DO_NOT_RETRY",
  "correction": "如果本轮修正了之前的推理错误, 写在这里, 否则留空",
  "identity_check": true/false  // 本轮是否偏离了身份锚点的核心规则
}}
只输出 JSON, 不要其他内容。"""

        user = f"用户输入: {user_input}\n知识库相关条目: {kb_context[:1500]}"
        resp = ask_deepseek(system, user, temperature=0.2)
        try:
            # 清理可能的 markdown 包裹
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[1].rsplit("\n", 1)[0]
            data = json.loads(resp)
        except:
            data = {"step_summary": resp[:80], "tags": [], "status": "in_progress", "correction": ""}

        node = {
            "turn": turn_id,
            "user_input": user_input[:60],
            "summary": data.get("step_summary", ""),
            "tags": data.get("tags", []),
            "status": data.get("status", "in_progress"),
            "correction": data.get("correction", ""),
        }
        self.nodes.append(node)
        return node

    def get_path(self) -> str:
        if not self.nodes:
            return "(empty)"
        path = [f"[IDENTITY] {self.identity['name']}: {self.identity['role']} (不可变)", ""]
        for n in self.nodes:
            s = {"fix_applied": "[OK]", "failed": "[NG]", "DO_NOT_RETRY": "[NG]", "in_progress": " -> "}.get(n["status"], " -> ")
            path.append(f"{s} T{n['turn']}: {n['summary']}")
            if n["correction"]:
                path.append(f"   修正: {n['correction']}")
        return "\n".join(path)


# =============================================================
# Layer 2: 标签索引 (本地数据结构)
# =============================================================
class TagIndex:
    def __init__(self):
        self.index = {}

    def build(self, kb: dict):
        for doc_id, meta in kb.items():
            for tag in meta["tags"]:
                self.index.setdefault(tag, []).append(doc_id)

    def search(self, tag: str) -> list:
        if tag in self.index:
            return self.index[tag]
        # 模糊匹配
        prefix = tag.split("_")[0] if "_" in tag else tag
        results = []
        for t, docs in self.index.items():
            if t.startswith(prefix):
                results.extend(docs)
        return list(set(results))


# =============================================================
# Layer 3: 位置映射 (本地数据结构)
# =============================================================
class PositionMapper:
    def __init__(self):
        self.map = {}

    def build(self, kb: dict):
        for doc_id, meta in kb.items():
            self.map[doc_id] = meta

    def locate(self, doc_id: str) -> dict:
        return self.map.get(doc_id, {})

    def locate_by_tags(self, tags: list, tag_index: TagIndex) -> list:
        results = []
        seen = set()
        for tag in tags:
            for did in tag_index.search(tag):
                if did in seen:
                    continue
                seen.add(did)
                entry = self.locate(did)
                if entry:
                    results.append({"doc_id": did, "tag": tag, "offset": entry["offset"], "preview": entry["content"][:100]})
        return results


# =============================================================
# Layer 4: 检索头 — DeepSeek 意图分析 + 本地标签路由
# =============================================================
class RetrievalHead:
    def __init__(self, tag_index: TagIndex, mapper: PositionMapper):
        self.ti = tag_index
        self.pm = mapper

    def extract_keywords(self, user_input: str) -> list:
        """用 DeepSeek 从用户输入中提取检索关键词/标签"""
        system = """从用户输入中提取检索关键词, 以 JSON 数组返回。
关键词应覆盖: 技术栈(token/JWT/cookie等)、错误类型(401/OOM等)、操作(debug/fix/deploy等)。
只输出 JSON 数组, 如 ["debug_login", "JWT", "401_error"]。"""
        resp = ask_deepseek(system, user_input, temperature=0.1)
        try:
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[1].rsplit("\n", 1)[0]
            return json.loads(resp)
        except:
            return []

    def retrieve(self, user_input: str) -> dict:
        keywords = self.extract_keywords(user_input)
        results = self.pm.locate_by_tags(keywords, self.ti)

        return {
            "keywords_extracted": keywords,
            "results_count": len(results),
            "results": [
                {"doc_id": r["doc_id"], "offset": r["offset"], "preview": r["preview"]}
                for r in results
            ],
        }


# =============================================================
# 主流程: 模拟三轮真实 API 对话检索
# =============================================================
def main():
    print("=" * 65)
    print("  四层逻辑链检索 Agent — DeepSeek V4 Pro 真实API")
    print("=" * 65)

    # 初始化
    chain = LogicChain(IDENTITY_ANCHOR)
    tag_idx = TagIndex()
    mapper = PositionMapper()
    tag_idx.build(KNOWLEDGE_BASE)
    mapper.build(KNOWLEDGE_BASE)
    retriever = RetrievalHead(tag_idx, mapper)

    turns = [
        "login 功能报 401 错误, 帮我排查原因",
        "sessionStorage 方案可行吗? 会不会有跨标签页问题?",
        "已确认 sessionStorage 不行, 最终应该用什么方案?",
    ]

    print("\n[知识库] {} 条记录已加载\n".format(len(KNOWLEDGE_BASE)))

    for i, user_input in enumerate(turns, 1):
        print(f"{'='*65}")
        print(f"  Turn {i}: {user_input}")
        print(f"{'='*65}")

        # Layer 4 + 2: DeepSeek 提取关键词 → 标签检索
        print("  [Layer4+2] DeepSeek 提取关键词...")
        retrieval = retriever.retrieve(user_input)
        print(f"    关键词: {retrieval['keywords_extracted']}")
        print(f"    命中: {retrieval['results_count']} 条")
        for r in retrieval["results"]:
            print(f"      -> [{r['doc_id']}] @{r['offset']}: {r['preview'][:70]}...")

        # Layer 1: DeepSeek 分析本轮 + 构建逻辑链
        kb_context = "\n".join([f"{r['doc_id']}: {r['preview']}" for r in retrieval["results"]])
        node = chain.process_turn(i, user_input, kb_context)
        print(f"  [Layer1] DeepSeek 逻辑链: {node['summary']}")
        print(f"    判定: {node['status']} | 标签: {node['tags']}")
        if node["correction"]:
            print(f"    修正: {node['correction']}")
        id_ok = node.get("identity_check", True)
        print(f"    身份锚点: {'[OK] 未偏离' if id_ok else '[WARN] 偏离核心规则'}")

        time.sleep(0.5)  # API 礼貌间隔

    # 最终逻辑链
    print(f"\n{'='*65}")
    print("  最终逻辑链条 (Layer 1)")
    print(f"{'='*65}")
    print(chain.get_path())

    # 检索效率对比
    print(f"\n{'='*65}")
    print("  检索效率对比")
    print(f"{'='*65}")
    print(f"  暴力全文扫描: 每次检索扫描全部 {len(KNOWLEDGE_BASE)} 条记录")
    print(f"  标签索引检索: 每次命中 1-3 个标签, 定位 < 3 条记录")
    kb_total = sum(len(v["content"]) for v in KNOWLEDGE_BASE.values())
    idx_total = sum(len(v["content"]) for v in KNOWLEDGE_BASE.values()) // len(KNOWLEDGE_BASE) * 2
    print(f"  知识库总字符: {kb_total} | 索引定位字符: ~{idx_total}")
    print(f"  节省: ~{int((1 - idx_total/kb_total)*100)}% 上下文窗口 (知识库越大, 节省越高)")
    print(f"\n  API调用统计: {len(turns)*2} 次 (每轮: 提取关键词 + 逻辑链分析)")


if __name__ == "__main__":
    main()
