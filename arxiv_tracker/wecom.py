# -*- coding: utf-8 -*-
"""企业微信群机器人推送。

支持两种消息类型：
  - text（默认）：纯文本 + 链接。个人微信的「微信插件」只支持 text 类型，
    markdown 会显示「暂不支持此消息类型」——所以个人微信场景必须用 text。
  - markdown：仅在企业微信 App 内查看时使用。

配置（config.yaml）:
    wecom:
      enabled: true
      webhook_env: "WECOM_WEBHOOK_URL"
      msg_type: "text"
      max_items: 5
      digest_max_chars: 130
"""
import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

# 企业微信消息体上限：markdown 4096 字节 / text 2048 字节（UTF-8），各留余量
_LIMITS = {"markdown": 3800, "text": 1900}


def _truncate(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _short_url(item: Dict[str, Any]) -> str:
    return item.get("html_url") or item.get("pdf_url") or (item.get("id") or "")


def _meta_line(item: Dict[str, Any], ex: Dict[str, Any]) -> str:
    """「匹配 92% · DeepMind · Stanford」样式的元信息行。"""
    bits = []
    interest = item.get("_interest")
    if interest is not None:
        bits.append(f"匹配 {round(float(interest) * 100)}%")
    affs = ex.get("affiliations") or item.get("_curator_insts") or []
    bits.extend(affs[:3])
    return " · ".join(bits)


def _pick_digest(item: Dict[str, Any], ex: Dict[str, Any],
                 summaries_zh: Dict, summaries_en: Dict,
                 digest_max_chars: int) -> str:
    sid = item.get("id") or ""
    s = summaries_zh.get(sid) or {}
    digest = ex.get("digest_zh") or s.get("digest_zh") or s.get("tldr") or ""
    if not digest:
        s_en = summaries_en.get(sid) or {}
        digest = s_en.get("digest_en") or s_en.get("tldr") or ""
    # 精选论文（有 enrich 结果）给 3 倍摘要额度
    return _truncate(digest, digest_max_chars * 3 if ex.get("digest_zh") else digest_max_chars)


def _render_item_text(idx: int, item: Dict[str, Any], ex: Dict[str, Any],
                      summaries_zh: Dict, summaries_en: Dict,
                      translations: Dict, digest_max_chars: int) -> str:
    sid = item.get("id") or ""
    tx = translations.get(sid) or {}
    title_zh = _truncate(ex.get("title_zh") or tx.get("title_zh") or "", 80)
    title_en = _truncate(item.get("title") or "", 120)

    lines = [f"{idx}. {title_zh or title_en}"]
    if title_zh:
        lines.append(f"   {title_en}")
    meta = _meta_line(item, ex)
    if meta:
        lines.append(f"   [{meta}]")
    digest = _pick_digest(item, ex, summaries_zh, summaries_en, digest_max_chars)
    if digest:
        lines.append(f"   {digest}")
    if ex.get("why_read"):
        lines.append(f"   💡 {_truncate(ex['why_read'], 100)}")
    lines.append(f"   {_short_url(item)}")
    return "\n".join(lines)


def _render_item_markdown(idx: int, item: Dict[str, Any], ex: Dict[str, Any],
                          summaries_zh: Dict, summaries_en: Dict,
                          translations: Dict, digest_max_chars: int) -> str:
    sid = item.get("id") or ""
    tx = translations.get(sid) or {}
    title = _truncate(item.get("title") or "", 160)
    title_zh = _truncate(ex.get("title_zh") or tx.get("title_zh") or "", 160)

    lines = [f"**{idx}. [{title}]({_short_url(item)})**"]
    meta = _meta_line(item, ex)
    if meta:
        lines.append(f"> {meta}")
    if title_zh:
        lines.append(f"> {title_zh}")
    digest = _pick_digest(item, ex, summaries_zh, summaries_en, digest_max_chars)
    if digest:
        lines.append(f"> {digest}")
    if ex.get("why_read"):
        lines.append(f"> 💡 {_truncate(ex['why_read'], 120)}")
    return "\n".join(lines)


def _split_batches(header: str, blocks: List[str], footer: str, byte_limit: int) -> List[str]:
    """按字节上限把条目切成多条消息，每条自带 header。"""
    batches, cur = [], header
    for b in blocks:
        candidate = cur + "\n\n" + b
        if len(candidate.encode("utf-8")) > byte_limit and cur != header:
            batches.append(cur)
            cur = header + "\n\n" + b
        else:
            cur = candidate
    if footer:
        candidate = cur + "\n\n" + footer
        if len(candidate.encode("utf-8")) <= byte_limit:
            cur = candidate
        else:
            batches.append(cur)
            cur = footer
    batches.append(cur)
    return batches


def build_wecom_messages(items: List[Dict[str, Any]],
                         summaries_zh: Optional[Dict] = None,
                         summaries_en: Optional[Dict] = None,
                         translations: Optional[Dict] = None,
                         max_items: int = 5,
                         digest_max_chars: int = 130,
                         site_url: str = "",
                         date_str: str = "",
                         extras: Optional[Dict[str, Dict]] = None,
                         header_title: str = "arXiv 论文速递",
                         msg_type: str = "text",
                         candidates_total: int = 0) -> List[str]:
    summaries_zh = summaries_zh or {}
    summaries_en = summaries_en or {}
    translations = translations or {}
    extras = extras or {}
    if not items:
        return []  # 无新增不打扰

    is_text = msg_type == "text"
    total = len(items)
    shown = items[:max_items]

    stat = f"本轮新增 {candidates_total} 篇，精选 {total} 篇" if candidates_total else f"共 {total} 篇"
    if total > max_items:
        stat += f"，推送前 {max_items} 篇"

    if is_text:
        header = f"📄 {header_title} {date_str}\n{stat}"
        footer = f"🔗 完整摘要与归档：\n{site_url}" if site_url else ""
        render = _render_item_text
    else:
        header = f"## 📄 {header_title}{('（' + date_str + '）') if date_str else ''}\n{stat}"
        footer = f"🔗 [完整摘要与历史归档]({site_url})" if site_url else ""
        render = _render_item_markdown

    blocks = [
        render(i, it, extras.get(it.get("id") or "") or {},
               summaries_zh, summaries_en, translations, digest_max_chars)
        for i, it in enumerate(shown, 1)
    ]
    return _split_batches(header, blocks, footer, _LIMITS["text" if is_text else "markdown"])


def send_wecom(webhook_url: str, content: str, msg_type: str = "text",
               timeout: int = 15) -> None:
    if msg_type == "text":
        payload = {"msgtype": "text", "text": {"content": content}}
    else:
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
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
                  header_title: str = "arXiv 论文速递",
                  candidates_total: int = 0) -> bool:
    """返回是否实际发送成功（无新增/未配置返回 False，不抛异常）。"""
    webhook_env = wecom_cfg.get("webhook_env") or "WECOM_WEBHOOK_URL"
    webhook_url = wecom_cfg.get("webhook_url") or os.getenv(webhook_env, "")
    if not webhook_url:
        print(f"[WeCom] 跳过：未找到 webhook（环境变量 {webhook_env} 未设置）")
        return False

    msg_type = str(wecom_cfg.get("msg_type", "text")).lower()
    messages = build_wecom_messages(
        items,
        summaries_zh=summaries_zh,
        summaries_en=summaries_en,
        translations=translations,
        max_items=int(wecom_cfg.get("max_items", 5)),
        digest_max_chars=int(wecom_cfg.get("digest_max_chars", 130)),
        site_url=site_url,
        date_str=date_str,
        extras=extras,
        header_title=header_title,
        msg_type=msg_type,
        candidates_total=candidates_total,
    )
    if not messages:
        print("[WeCom] 今日无新增，跳过推送")
        return False

    for i, msg in enumerate(messages, 1):
        send_wecom(webhook_url, msg, msg_type=msg_type)
        print(f"[WeCom] 已发送 {i}/{len(messages)} ({msg_type})")
    return True
