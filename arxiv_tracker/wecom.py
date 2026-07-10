# -*- coding: utf-8 -*-
"""企业微信群机器人推送。

将每日 arXiv 摘要以 markdown 消息推送到企业微信群机器人 webhook。
配合「微信插件」绑定后，消息可直接在个人微信中查看。

配置（config.yaml）:
    wecom:
      enabled: true
      webhook_env: "WECOM_WEBHOOK_URL"   # webhook 从环境变量读取，勿写入仓库
      max_items: 15                      # 单次推送最多条目数
      digest_max_chars: 120              # 每篇摘要截断长度
"""
import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

# 企业微信 markdown 消息上限 4096 字节（UTF-8），留余量避免边界被拒
_BYTE_LIMIT = 3800


def _truncate(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _render_item(idx: int, item: Dict[str, Any],
                 summaries_zh: Dict[str, Dict[str, str]],
                 summaries_en: Dict[str, Dict[str, str]],
                 translations: Dict[str, Dict[str, str]],
                 digest_max_chars: int,
                 extras: Optional[Dict[str, Dict]] = None) -> str:
    sid = item.get("id") or ""
    title = _truncate(item.get("title") or "", 160)
    url = item.get("html_url") or item.get("pdf_url") or sid
    tx = translations.get(sid) or {}
    ex = (extras or {}).get(sid) or {}
    title_zh = _truncate(ex.get("title_zh") or tx.get("title_zh") or "", 160)

    s = summaries_zh.get(sid) or {}
    digest = ex.get("digest_zh") or s.get("digest_zh") or s.get("tldr") or ""
    if not digest:
        s_en = summaries_en.get(sid) or {}
        digest = s_en.get("digest_en") or s_en.get("tldr") or ""
    # 入选精读论文给更长的摘要额度
    digest = _truncate(digest, digest_max_chars * 3 if ex else digest_max_chars)

    lines = [f"**{idx}. [{title}]({url})**"]
    meta_bits = []
    if ex.get("affiliations"):
        meta_bits.append("🏛 " + " · ".join(ex["affiliations"]))
    score = item.get("_curator_score")
    if score is not None:
        meta_bits.append(f"⭐ {score:.2f}")
    if meta_bits:
        lines.append("> " + "　".join(meta_bits))
    if title_zh:
        lines.append(f"> {title_zh}")
    if digest:
        lines.append(f"> {digest}")
    if ex.get("why_read"):
        lines.append(f"> 💡 {_truncate(ex['why_read'], 120)}")
    return "\n".join(lines)


def _split_batches(header: str, blocks: List[str], footer: str = "") -> List[str]:
    """按字节上限把条目切成多条消息，每条自带 header。"""
    batches, cur = [], header
    for b in blocks:
        candidate = cur + "\n\n" + b
        if len(candidate.encode("utf-8")) > _BYTE_LIMIT and cur != header:
            batches.append(cur)
            cur = header + "\n\n" + b
        else:
            cur = candidate
    if footer:
        candidate = cur + "\n\n" + footer
        if len(candidate.encode("utf-8")) <= _BYTE_LIMIT:
            cur = candidate
    batches.append(cur)
    return batches


def build_wecom_messages(items: List[Dict[str, Any]],
                         summaries_zh: Optional[Dict] = None,
                         summaries_en: Optional[Dict] = None,
                         translations: Optional[Dict] = None,
                         max_items: int = 15,
                         digest_max_chars: int = 120,
                         site_url: str = "",
                         date_str: str = "",
                         extras: Optional[Dict[str, Dict]] = None,
                         header_title: str = "arXiv 论文速递") -> List[str]:
    summaries_zh = summaries_zh or {}
    summaries_en = summaries_en or {}
    translations = translations or {}

    header = f"## 📄 {header_title}{('（' + date_str + '）') if date_str else ''}"
    if not items:
        return []  # 无新增不打扰

    total = len(items)
    shown = items[:max_items]
    header += f"\n共 {total} 篇" + (f"，本条推送前 {max_items} 篇" if total > max_items else "")

    blocks = [
        _render_item(i, it, summaries_zh, summaries_en, translations,
                     digest_max_chars, extras=extras)
        for i, it in enumerate(shown, 1)
    ]
    footer = f"🔗 [完整摘要与历史归档]({site_url})" if site_url else ""
    return _split_batches(header, blocks, footer)


def send_wecom(webhook_url: str, markdown_text: str, timeout: int = 15) -> None:
    payload = {"msgtype": "markdown", "markdown": {"content": markdown_text}}
    resp = requests.post(webhook_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errcode") != 0:
        raise RuntimeError(f"WeCom API error: {json.dumps(data, ensure_ascii=False)}")


def push_to_wecom(items: List[Dict[str, Any]],
                  wecom_cfg: Dict[str, Any],
                  summaries_zh: Optional[Dict] = None,
                  summaries_en: Optional[Dict] = None,
                  translations: Optional[Dict] = None,
                  site_url: str = "",
                  date_str: str = "",
                  extras: Optional[Dict[str, Dict]] = None,
                  header_title: str = "arXiv 论文速递") -> bool:
    """返回是否实际发送成功（无新增/未配置返回 False，不抛异常）。"""
    webhook_env = wecom_cfg.get("webhook_env") or "WECOM_WEBHOOK_URL"
    webhook_url = wecom_cfg.get("webhook_url") or os.getenv(webhook_env, "")
    if not webhook_url:
        print(f"[WeCom] 跳过：未找到 webhook（环境变量 {webhook_env} 未设置）")
        return False

    messages = build_wecom_messages(
        items,
        summaries_zh=summaries_zh,
        summaries_en=summaries_en,
        translations=translations,
        max_items=int(wecom_cfg.get("max_items", 15)),
        digest_max_chars=int(wecom_cfg.get("digest_max_chars", 120)),
        site_url=site_url,
        date_str=date_str,
        extras=extras,
        header_title=header_title,
    )
    if not messages:
        print("[WeCom] 今日无新增，跳过推送")
        return False

    for i, msg in enumerate(messages, 1):
        send_wecom(webhook_url, msg)
        print(f"[WeCom] 已发送 {i}/{len(messages)}")
    return True
