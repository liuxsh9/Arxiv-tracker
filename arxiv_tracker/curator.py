# -*- coding: utf-8 -*-
"""兴趣点筛选 + 机构择优 + 精读级摘要（curator）。

流水线（在去重后的候选集上执行）：
  1. LLM 批量兴趣打分（title+abstract → 0~1 分），低于 min_score 的丢弃
  2. 抓取 arXiv HTML 版页头，正则匹配 top_institutions 关键词（不耗 token）
  3. rank = 兴趣分 + 机构加分，排序取 top max_push
  4. 仅对入选论文做一次丰富版总结（机构/中文标题/摘要/是否值得读原文）

配置（config.yaml）:
    curator:
      enabled: true
      min_score: 0.6
      max_push: 5
      inst_bonus: 0.15
      interests: |
        - 你的兴趣点，每行一条
      top_institutions: ["DeepMind", "OpenAI", ...]
"""
import re
import json
import requests
from typing import Any, Dict, List, Optional, Tuple

from .llm import _chat_completions_request, _json_loose

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", re.I)


def _arxiv_id(item: Dict[str, Any]) -> str:
    """从 abs URL / id 提取 2507.12345v1 形式的 ID。"""
    src = item.get("html_url") or item.get("id") or ""
    m = re.search(r"(?:abs|html)/([0-9]{4}\.[0-9]{4,6}(?:v\d+)?)", src)
    return m.group(1) if m else ""


def fetch_author_block(item: Dict[str, Any], timeout: int = 10) -> str:
    """抓 arXiv HTML 版（LaTeXML）页头文本，通常含作者与机构。

    没有 HTML 版（老论文/转换失败）时返回空串，不抛异常。
    """
    aid = _arxiv_id(item)
    if not aid:
        return ""
    try:
        r = requests.get(
            f"https://arxiv.org/html/{aid}",
            headers={"User-Agent": UA, "Accept": "text/html"},
            timeout=timeout, allow_redirects=True,
        )
        if r.status_code != 200:
            return ""
        # 只取正文前一段：标题/作者/机构都在页头
        body = r.text
        m = re.search(r"<body[^>]*>", body, re.I)
        if m:
            body = body[m.end():]
        text = _TAG_RE.sub(" ", body[:20000])
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1500]
    except Exception:
        return ""


def match_institutions(author_block: str, top_institutions: List[str]) -> List[str]:
    """在页头文本中做机构关键词匹配（大小写不敏感），返回命中列表。"""
    if not author_block:
        return []
    low = author_block.lower()
    return [inst for inst in (top_institutions or []) if inst.lower() in low]


def score_interest(items: List[Dict[str, Any]], interests: str,
                   llm_cfg: Dict[str, Any], api_key: str,
                   batch_size: int = 20, verbose: bool = False) -> Dict[str, float]:
    """批量兴趣打分。返回 {item_id: score}；解析失败的条目默认 0.5（宁可多看不漏）。"""
    scores: Dict[str, float] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        payload = [{
            "i": idx,
            "title": (it.get("title") or "")[:200],
            "abstract": (it.get("summary") or "")[:600],
        } for idx, it in enumerate(batch)]
        messages = [
            {"role": "system", "content":
                "You are a research-paper triage assistant. Score how well each paper "
                "matches the user's research interests. Be strict: only clearly on-topic "
                "papers deserve >0.7; tangential mentions deserve <0.4."},
            {"role": "user", "content":
                f"USER INTERESTS:\n{interests}\n\n"
                "PAPERS (JSON list):\n" + json.dumps(payload, ensure_ascii=False) + "\n\n"
                'Return STRICT JSON: {"scores": [{"i": <index>, "s": <0.0-1.0>}, ...]} '
                "covering every index. No commentary."},
        ]
        try:
            text = _chat_completions_request(
                base_url=llm_cfg.get("base_url", ""), api_key=api_key,
                model=llm_cfg.get("model", ""), messages=messages,
                temperature=0.0, max_tokens=1000,
            )
            data = _json_loose(text)
            got = {int(x["i"]): float(x["s"]) for x in (data.get("scores") or [])
                   if isinstance(x, dict) and "i" in x and "s" in x}
        except Exception as e:
            if verbose:
                print(f"[Curator] 兴趣打分批次失败（{e}），该批默认 0.5")
            got = {}
        for idx, it in enumerate(batch):
            sid = it.get("id") or ""
            scores[sid] = max(0.0, min(1.0, got.get(idx, 0.5)))
    return scores


def enrich_paper(item: Dict[str, Any], author_block: str,
                 llm_cfg: Dict[str, Any], api_key: str) -> Dict[str, str]:
    """对入选论文做一次丰富版提炼（单次 LLM 调用）。

    返回 {affiliations, title_zh, digest_zh, digest_en, why_read}
    digest_zh 3~5 句：动机 / 方法 / 关键结果（带数字）/ 亮点或局限。
    why_read 一句话：适合谁读、为何值得（或不值得）精读。
    """
    meta = {
        "title": item.get("title") or "",
        "abstract": (item.get("summary") or "")[:1800],
        "comments": item.get("comments") or "",
        "author_page_header": author_block[:800],
    }
    messages = [
        {"role": "system", "content":
            "You are a senior research assistant helping the user decide whether a paper "
            "is worth reading in full. Be concrete and objective; include key numbers "
            "from the abstract when available; no marketing language."},
        {"role": "user", "content":
            "Based on the metadata below, return STRICT JSON with keys:\n"
            '- "affiliations": array of up to 4 institution names extracted from '
            'author_page_header (English, normalized, e.g. "DeepMind"); [] if unknown\n'
            '- "title_zh": Simplified Chinese translation of the title\n'
            '- "digest_zh": 3-5 Chinese sentences covering motivation, method, key '
            "quantitative results, and one highlight or limitation\n"
            '- "digest_en": one concise English paragraph (motivation/method/results)\n'
            '- "why_read": ONE Chinese sentence telling the user who should read the '
            "full paper and why (or why a skim suffices)\n\n"
            "DATA:\n" + json.dumps(meta, ensure_ascii=False)},
    ]
    text = _chat_completions_request(
        base_url=llm_cfg.get("base_url", ""), api_key=api_key,
        model=llm_cfg.get("model", ""), messages=messages,
        temperature=0.2, max_tokens=900,
    )
    data = _json_loose(text)
    affs = data.get("affiliations") or []
    if not isinstance(affs, list):
        affs = [affs]

    def _as_str(v) -> str:
        # LLM 偶尔把字符串字段返回成句子数组，做类型容错
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        return str(v or "").strip()

    return {
        "affiliations": [str(a).strip() for a in affs if str(a).strip()][:4],
        "title_zh": _as_str(data.get("title_zh")),
        "digest_zh": _as_str(data.get("digest_zh")),
        "digest_en": _as_str(data.get("digest_en")),
        "why_read": _as_str(data.get("why_read")),
    }


def curate(items: List[Dict[str, Any]], curator_cfg: Dict[str, Any],
           llm_cfg: Dict[str, Any], api_key: str,
           verbose: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
    """执行完整 curation。返回 (入选 items, {id: enrich 结果})。

    入选 items 会带上 _curator_score / _curator_insts 字段供渲染层使用。
    """
    min_score = float(curator_cfg.get("min_score", 0.6))
    max_push = int(curator_cfg.get("max_push", 5))
    inst_bonus = float(curator_cfg.get("inst_bonus", 0.15))
    interests = (curator_cfg.get("interests") or "").strip()
    top_insts = curator_cfg.get("top_institutions") or []
    if not interests:
        print("[Curator] 跳过：config.yaml 的 curator.interests 为空")
        return items, {}

    # 1) 兴趣打分 + 阈值过滤
    scores = score_interest(items, interests, llm_cfg, api_key, verbose=verbose)
    passing = [it for it in items if scores.get(it.get("id") or "", 0) >= min_score]
    print(f"[Curator] 兴趣过滤: {len(items)} -> {len(passing)} (min_score={min_score})")
    if not passing:
        return [], {}

    # 2) 机构匹配（仅对过滤后的候选抓页头，纯 HTTP + 关键词，无 token 成本）
    blocks: Dict[str, str] = {}
    for it in passing:
        sid = it.get("id") or ""
        block = fetch_author_block(it) if curator_cfg.get("fetch_affiliations", True) else ""
        blocks[sid] = block
        insts = match_institutions(block, top_insts)
        it["_curator_insts"] = insts
        it["_interest"] = scores.get(sid, 0)          # 展示用：纯兴趣分（0~1）
        it["_curator_score"] = it["_interest"] + (inst_bonus if insts else 0)  # 排序用

    # 3) 排序择优
    passing.sort(key=lambda x: x.get("_curator_score", 0), reverse=True)
    selected = passing[:max_push]
    dropped = len(passing) - len(selected)
    if dropped > 0:
        print(f"[Curator] 择优推送 top {len(selected)}，{dropped} 篇过阈值但未入选（已归档到网页）")

    # 4) 丰富版提炼（仅入选论文，每篇 1 次调用；失败重试 1 次）
    extras: Dict[str, Dict] = {}
    for it in selected:
        sid = it.get("id") or ""
        for attempt in (1, 2):
            try:
                ex = enrich_paper(it, blocks.get(sid, ""), llm_cfg, api_key)
                if not (ex.get("digest_zh") or ex.get("why_read")) and attempt == 1:
                    continue  # 空响应也视为失败，重试一次
                # LLM 提取的机构 与 关键词命中 合并去重，命中的排前
                merged, seen = [], set()
                for a in (it.get("_curator_insts") or []) + ex["affiliations"]:
                    if a.lower() not in seen:
                        merged.append(a)
                        seen.add(a.lower())
                ex["affiliations"] = merged[:4]
                extras[sid] = ex
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[Curator] 提炼失败（已重试） {sid[:32]}: {e}")
    return selected, extras
