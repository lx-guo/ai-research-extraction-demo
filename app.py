from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from adapters.demo_repository import flatten_ids, load_demo, property_record


st.set_page_config(
    page_title="AI 科研文献智能抽取助手",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#172033; --muted:#667085; --line:#e6e9ef; --blue:#315efb; --blue-soft:#eef3ff; --green:#0f8a5f; }
      .stApp { background:#f5f7fa; color:var(--ink); }
      [data-testid="stHeader"] { display:none; }
      [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
      .block-container { padding-top:1.6rem; padding-bottom:2rem; max-width:1100px; }
      h1,h2,h3 { letter-spacing:-0.02em; color:var(--ink); }
      h1 { font-size:1.75rem !important; margin-bottom:.15rem !important; }
      h2 { font-size:1.25rem !important; }
      .product-kicker { color:#315efb; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      .subtitle { color:var(--muted); margin-bottom:1.2rem; }
      .topbar { display:flex; align-items:center; justify-content:space-between; padding:14px 18px; background:#fff; border:1px solid var(--line); border-radius:14px; margin-bottom:18px; }
      .paper-title { font-weight:700; font-size:1.02rem; }
      .paper-meta { color:var(--muted); font-size:.82rem; margin-top:3px; }
      .badge { display:inline-block; border-radius:999px; padding:5px 10px; font-size:.74rem; font-weight:700; background:var(--blue-soft); color:#2448c8; border:1px solid #d9e2ff; }
      .badge-green { background:#ecf9f3; color:#08734e; border-color:#cdeedf; }
      .badge-amber { background:#fff7e8; color:#985d00; border-color:#f8dfae; }
      .card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 1px 2px rgba(16,24,40,.03); }
      .metric-label { color:var(--muted); font-size:.76rem; margin-bottom:5px; }
      .metric-value { font-size:1.55rem; font-weight:750; color:var(--ink); }
      .metric-note { color:#98a2b3; font-size:.73rem; margin-top:3px; }
      .step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 12px; min-height:112px; }
      .step-index { color:#315efb; font-weight:800; font-size:.72rem; }
      .step-title { font-size:.95rem; font-weight:720; margin:9px 0 4px; }
      .step-status { color:#0f8a5f; font-size:.76rem; font-weight:650; }
      .property-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; min-height:122px; }
      .property-name { color:var(--muted); font-size:.76rem; font-weight:650; }
      .property-value { color:var(--ink); font-size:1.16rem; font-weight:740; margin:8px 0; overflow-wrap:anywhere; }
      .trace-arrow { color:#98a2b3; font-size:1.5rem; text-align:center; padding-top:34px; }
      .trace-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; min-height:116px; }
      .trace-label { color:#667085; font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
      .trace-main { font-size:1rem; font-weight:700; margin-top:9px; }
      .prototype { border:1px dashed #b9c2d0; background:#fafbfc; border-radius:12px; padding:15px; }
      .source { border-left:3px solid #315efb; background:#fff; border-radius:0 12px 12px 0; padding:16px 18px; color:#344054; line-height:1.72; }
      div[data-testid="stDataFrame"] { background:#fff; border-radius:12px; }
      .stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:650; }
      .stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] { background:#315efb; border-color:#315efb; }
      [data-testid="stMetricValue"] { font-size:1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGES = {
    "task": "01  创建任务",
    "pipeline": "02  处理流程",
    "samples": "03  样品识别",
    "results": "04  抽取结果",
    "evidence": "05  证据追溯",
    "export": "06  最终导出",
}

USER_STAGE_MAP = [
    ("01", "文档解析", ["stage01_blocks"]),
    ("02", "样品识别", ["stage02_keep", "stage03_sample_pack", "stage04_sample_discovery"]),
    ("03", "证据检索", ["stage05_searchtext", "stage06_retrieval"]),
    ("04", "性质抽取", ["stage07_extract_properties"]),
    ("05", "结果整理", ["stage08_finalize"]),
]

PROPERTY_LABELS = {
    "composition": "组成",
    "Mn": "分子量",
    "Tg": "热转变",
    "Hm": "熔融焓 / 结晶",
    "yb": "力学性能",
    "MI": "熔指 / 流变",
    "old": "老化 / 降解",
}


def query_value(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


def go(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def fmt_seconds(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 1:
        return "< 1 秒"
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes} 分 {remainder:02d} 秒"


def badge(text: str, tone: str = "blue") -> str:
    class_name = "badge"
    if tone == "green":
        class_name += " badge-green"
    elif tone == "amber":
        class_name += " badge-amber"
    return f'<span class="{class_name}">{html.escape(str(text))}</span>'


def status_badge(status: str | None) -> str:
    labels = {"ok": "已通过校验", "review": "建议复核", "missing": "未抽取到", "rejected": "已拒绝"}
    if status == "ok":
        return badge(labels[status], "green")
    if status in {"review", "rejected"}:
        return badge(labels.get(status, status), "amber")
    # 面向用户统一表达为待人工复核；真实的缺少专项 validator 原因仍保留在技术结果中。
    return badge(labels.get(status or "", "待人工复核"), "amber")


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="product-kicker">AI 科研抽取 · 证据优先</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def topbar(meta: dict[str, Any]) -> None:
    st.markdown(
        '<div class="topbar"><div><div class="paper-title">Polymer_Paper_01.docx</div>'
        f'<div class="paper-meta">缓存任务 DEMO-RUN-01 · {meta["final_sample_count"]} 个正式样品 · '
        f'{meta["property_group_count"]} 个性质组</div></div>'
        '<div><span class="badge badge-green">真实缓存结果</span> &nbsp; '
        '<span class="badge">演示模式</span></div></div>',
        unsafe_allow_html=True,
    )


def summarize_composition(value: Any) -> str:
    if not isinstance(value, dict):
        return "—"
    monomers = value.get("monomers") or []
    parts = []
    for monomer in monomers:
        name = monomer.get("abbrev") or monomer.get("name_en") or monomer.get("name_cn")
        ratio = monomer.get("composition_ratio")
        if name:
            parts.append(f"{name} {ratio:g}" if isinstance(ratio, (int, float)) else str(name))
    return " · ".join(parts) or "—"


def display_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, dict):
        if "monomers" in value:
            return summarize_composition(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " · ".join(str(item) for item in value)
    return str(value)


def result_cards(result: dict[str, Any]) -> None:
    raw = result.get("properties_raw") or {}
    cards = [
        ("组成", display_value(result.get("composition")), (raw.get("composition") or {}).get("validation_status")),
        ("Mn / Mw / PDI", " / ".join(display_value(result.get(key)) for key in ("Mn", "Mw", "PDI")), (raw.get("Mn") or {}).get("validation_status")),
        ("Tg / Tm / Tc", " / ".join(display_value(result.get(key)) for key in ("Tg", "Tm", "Tc")), (raw.get("Tg") or {}).get("validation_status")),
        ("Hm", display_value(result.get("Hm")), (raw.get("Hm") or {}).get("validation_status")),
        ("力学性能", display_value(result.get("yb") or result.get("tensile_strength")), (raw.get("yb") or {}).get("validation_status")),
        ("老化 / 降解", display_value(result.get("aging_rate")), (raw.get("old") or {}).get("validation_status")),
    ]
    for row in range(2):
        columns = st.columns(3)
        for column, item in zip(columns, cards[row * 3 : row * 3 + 3]):
            label, value, status = item
            with column:
                st.markdown(
                    f'<div class="property-card"><div class="property-name">{html.escape(label)}</div>'
                    f'<div class="property-value">{html.escape(value)}</div>{status_badge(status)}</div>',
                    unsafe_allow_html=True,
                )


def selected_result(data: dict[str, Any]) -> dict[str, Any]:
    results = data["results"]
    requested = query_value("sample", "S1")
    ids = [str(item.get("sample_id")) for item in results]
    if requested not in ids:
        requested = ids[0]
    label_map = {str(item.get("sample_id")): f'{item.get("sample_id")} · {item.get("sample_name")}' for item in results}
    chosen = st.selectbox(
        "选择样品",
        ids,
        index=ids.index(requested),
        format_func=lambda item: label_map[item],
    )
    return next(item for item in results if str(item.get("sample_id")) == chosen)


def render_task(data: dict[str, Any]) -> None:
    page_header("创建抽取任务", "上传论文，或直接体验一条已完成的真实缓存任务。")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.text_input("论文名称", value="Polymer_Paper_01.docx", disabled=True)
        st.file_uploader("上传 DOCX", type=["docx"], disabled=True, help="公开 Demo 使用真实匿名化缓存，不读取上传文件。")
        st.caption("公开体验不调用 LLM，不需要 API Key。")
        if st.button("使用示例论文体验", type="primary", use_container_width=True):
            go("pipeline")
    with right:
        st.markdown("### 示例任务概览")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("原始 block", f'{data["meta"]["block_count"]}', "段落 / 表格 / 标题")
        with c2:
            metric_card("正式样品", f'{data["meta"]["formal_sample_count"]}', "来自 36 个候选")
        st.write("")
        c3, c4 = st.columns(2)
        with c3:
            metric_card("性质组", f'{data["meta"]["property_group_count"]}', "真实 Pipeline 配置")
        with c4:
            metric_card("Evidence", f'{data["meta"]["unique_evidence_id_count"]}', "唯一 block ID")
        st.info("示例论文已做基础匿名化；科学数值未改写，身份类路径、姓名与 DOI 已脱敏。")


def render_pipeline(data: dict[str, Any]) -> None:
    page_header("处理流程", "用业务语言理解 AI 如何从文档走到可核验结果。")
    topbar(data["meta"])
    status_by_stage = {item["stage"]: item for item in data["meta"]["stages"]}
    columns = st.columns(5)
    for column, (number, title, stages) in zip(columns, USER_STAGE_MAP):
        seconds = sum(status_by_stage[stage]["duration_seconds"] for stage in stages)
        with column:
            st.markdown(
                f'<div class="step"><div class="step-index">STEP {number}</div>'
                f'<div class="step-title">{title}</div><div class="step-status">✓ 已完成</div>'
                f'<div class="metric-note">{fmt_seconds(seconds)}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    a, b, c, d = st.columns(4)
    with a:
        metric_card("处理状态", "8 / 8", "全部 stage 完成")
    with b:
        metric_card("检索模式", "按性质检索", "每个性质 50 个候选 block")
    with c:
        metric_card("性质抽取调用", str(data["meta"]["stage07_api_call_count"]), "缓存 run 实际记录")
    with d:
        metric_card("批次执行", "21 / 21", "本次性质抽取 batch 全部成功")
    with st.expander("技术详情 · stage / status / duration"):
        rows = [
            {
                "技术阶段": item["stage"],
                "状态": item["status"],
                "耗时": fmt_seconds(item["duration_seconds"]),
                "开始": item["started_at"],
                "结束": item["finished_at"],
            }
            for item in data["meta"]["stages"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_samples(data: dict[str, Any]) -> None:
    page_header("样品识别", "先建立样品目录，再把后续抽取结果绑定到明确的样品。")
    topbar(data["meta"])
    c1, c2, c3, c4 = st.columns(4)
    counts = pd.Series([sample.get("sample_quality_status") for sample in data["samples"]]).value_counts().to_dict()
    with c1:
        metric_card("候选样品", str(len(data["samples"])), "sample_merged.json")
    with c2:
        metric_card("进入正式抽取", str(len(data["formal_samples"])), "quality_status = keep")
    with c3:
        metric_card("建议移除", str(counts.get("remove", 0)), "规则后处理结果")
    with c4:
        metric_card("建议复核", str(counts.get("review", 0)), "需要人工判断")
    st.write("")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        rows = [
            {
                "ID": sample.get("sample_id"),
                "样品名称": sample.get("sample_name"),
                "阶段": sample.get("polymer_stage") or "—",
                "范围": sample.get("scope") or "—",
                "证据数": len(sample.get("evidence_block_id") or []),
                "状态": sample.get("sample_quality_status"),
            }
            for sample in data["formal_samples"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=390)
    with right:
        ids = [str(item.get("sample_id")) for item in data["formal_samples"]]
        chosen = st.selectbox("查看样品", ids, format_func=lambda sid: f'{sid} · {next(s["sample_name"] for s in data["formal_samples"] if s["sample_id"] == sid)}')
        sample = next(item for item in data["formal_samples"] if str(item.get("sample_id")) == chosen)
        st.markdown(f"### {sample.get('sample_name')}")
        st.caption(f"{sample.get('sample_id')} · {sample.get('polymer_stage') or '未标注阶段'}")
        st.write(sample.get("definition_text") or "暂无定义文本")
        st.markdown("**样品发现 Evidence ID**")
        st.code(" · ".join(sample.get("evidence_block_id") or []) or "无", language=None)
        st.markdown(
            '<div class="prototype"><b>评审动作 · 原型交互</b><br>'
            '<span style="color:#667085;font-size:.82rem">确认 / 删除 / 合并仅为交互原型，不写回现有后端。</span></div>',
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        b1.button("确认", disabled=True, use_container_width=True)
        b2.button("删除", disabled=True, use_container_width=True)
        b3.button("合并", disabled=True, use_container_width=True)


def render_results(data: dict[str, Any]) -> None:
    page_header("抽取结果", "按“样品 × 属性”查看结果，并保留校验状态与原始技术输出。")
    topbar(data["meta"])
    result = selected_result(data)
    st.markdown(f"### {html.escape(str(result.get('sample_name')))}")
    st.caption(f"{result.get('sample_id')} · {result.get('polymer_stage') or '未标注阶段'} · sample_quality_status: {result.get('sample_quality_status')}")
    result_cards(result)
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1.15])
    with c1:
        st.info("31 个样品均有 Mn / Mw / PDI 结果；其中 24 条通过专项校验，7 条建议复核。")
    with c2:
        st.info("16 个样品具有通过校验的 Tg / Tm / Tc；其余 15 个标记为未抽取到。")
    with c3:
        st.warning("问题案例：Hm 组存在字段错配现象，演示界面保留原结果并明确提示人工复核。")
    with st.expander("查看技术结果 · properties_raw"):
        st.json(result.get("properties_raw") or {})


def output_summary(result: dict[str, Any], property_name: str) -> str:
    if property_name == "composition":
        return summarize_composition(result.get("composition"))
    if property_name == "Mn":
        return " / ".join(display_value(result.get(key)) for key in ("Mn", "Mw", "PDI"))
    if property_name == "Tg":
        return " / ".join(display_value(result.get(key)) for key in ("Tg", "Tm", "Tc"))
    if property_name == "Hm":
        return display_value(result.get("Hm"))
    if property_name == "yb":
        return display_value(result.get("yb") or result.get("tensile_strength") or result.get("elongation_at_break"))
    if property_name == "MI":
        return display_value(result.get("MI") or result.get("relaxation_time") or result.get("rheology"))
    return display_value(result.get("aging_rate") or result.get("old"))


def render_block(block: dict[str, Any]) -> None:
    heading = " › ".join(block.get("heading_path") or [])
    st.markdown(f"#### {html.escape(str(block.get('block_id')))}")
    st.caption(f"{block.get('type')} · doc_pos {block.get('doc_pos')}" + (f" · {heading}" if heading else ""))
    content = block.get("content")
    if isinstance(content, dict) and content.get("rows") is not None:
        caption = content.get("caption") or content.get("table_id") or "表格"
        st.markdown(f"**{html.escape(str(caption))}**")
        columns = content.get("columns") or []
        rows = content.get("rows") or []
        width = max([len(columns), *[len(row) for row in rows]] or [0])
        source_columns = list(columns) + [f"字段 {index + 1}" for index in range(len(columns), width)]
        normalized_columns = []
        seen_columns: dict[str, int] = {}
        for index, column in enumerate(source_columns):
            base = str(column).strip() or f"未命名字段 {index + 1}"
            seen_columns[base] = seen_columns.get(base, 0) + 1
            normalized_columns.append(base if seen_columns[base] == 1 else f"{base} ({seen_columns[base]})")
        normalized_rows = [list(row) + [None] * (width - len(row)) for row in rows]
        if width:
            st.dataframe(pd.DataFrame(normalized_rows, columns=normalized_columns), hide_index=True, use_container_width=True)
    else:
        st.markdown(f'<div class="source">{html.escape(display_value(content))}</div>', unsafe_allow_html=True)
    span = block.get("source_span") or {}
    if span:
        st.caption("原始定位字段：" + " · ".join(f"{key}={value}" for key, value in span.items()))


def render_evidence(data: dict[str, Any]) -> None:
    page_header("证据追溯", "从 AI 输出回到证据块，再查看论文中的真实原始内容。")
    topbar(data["meta"])
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        result = selected_result(data)
    with right:
        requested_property = query_value("property", "Mn")
        property_names = list(PROPERTY_LABELS)
        if requested_property not in property_names:
            requested_property = "Mn"
        property_name = st.selectbox(
            "选择属性",
            property_names,
            index=property_names.index(requested_property),
            format_func=lambda name: f"{name} · {PROPERTY_LABELS[name]}",
        )
    record = property_record(result, property_name)
    evidence_ids = flatten_ids(record)
    blocks = [data["blocks_by_id"][item] for item in evidence_ids if item in data["blocks_by_id"]]
    status = record.get("validation_status")
    c1, arrow1, c2, arrow2, c3 = st.columns([1.2, .22, 1.15, .22, 1.35])
    with c1:
        st.markdown(
            f'<div class="trace-card"><div class="trace-label">抽取结果</div>'
            f'<div class="trace-main">{html.escape(output_summary(result, property_name))}</div>'
            f'<div style="margin-top:8px">{status_badge(status)}</div></div>',
            unsafe_allow_html=True,
        )
    with arrow1:
        st.markdown('<div class="trace-arrow">→</div>', unsafe_allow_html=True)
    with c2:
        ids_text = "<br>".join(html.escape(item) for item in evidence_ids[:3]) or "未记录 Evidence ID"
        st.markdown(
            f'<div class="trace-card"><div class="trace-label">证据块</div>'
            f'<div class="trace-main" style="font-size:.82rem">{ids_text}</div>'
            f'<div class="metric-note">共 {len(evidence_ids)} 个唯一 ID</div></div>',
            unsafe_allow_html=True,
        )
    with arrow2:
        st.markdown('<div class="trace-arrow">→</div>', unsafe_allow_html=True)
    with c3:
        first = blocks[0] if blocks else {}
        st.markdown(
            f'<div class="trace-card"><div class="trace-label">原文内容</div>'
            f'<div class="trace-main">{html.escape(str(first.get("type") or "不可用"))} · '
            f'doc_pos {html.escape(str(first.get("doc_pos") or "—"))}</div>'
            '<div class="metric-note">无页码；展示系统真实定位字段</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    if not blocks:
        st.warning("该属性记录没有可解析的 Evidence block ID。")
    else:
        labels = [f'{block.get("block_id")} · {block.get("type")}' for block in blocks]
        chosen_label = st.selectbox("查看原始证据", labels)
        render_block(blocks[labels.index(chosen_label)])
    with st.expander("查看属性技术结果"):
        st.json(record)


def render_export(data: dict[str, Any]) -> None:
    page_header("最终导出", "预览结构化结果，并下载真实缓存产物。")
    topbar(data["meta"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("样品记录", str(len(data["results"])), "final/result.json")
    with c2:
        metric_card("性质组", str(data["meta"]["property_group_count"]), "all_properties.json")
    with c3:
        metric_card("Evidence 可解析", f'{data["meta"]["evidence_resolution_count"]} / {data["meta"]["unique_evidence_id_count"]}', "唯一 block ID")
    with c4:
        metric_card("导出格式", "JSON + TXT", "原 Pipeline 已生成")
    st.write("")
    st.info("部分性质尚待专项校验。导出结果保留真实抽取值，请结合 Evidence 完成复核后使用。")
    preview = []
    for row in data["results"][:8]:
        preview.append(
            {
                "样品": row.get("sample_name"),
                "Mn": row.get("Mn"),
                "Mw": row.get("Mw"),
                "PDI": row.get("PDI"),
                "Tg": row.get("Tg"),
                "Tm": row.get("Tm"),
                "Hm": row.get("Hm"),
            }
        )
    st.markdown("### 结果预览")
    st.dataframe(pd.DataFrame(preview), hide_index=True, use_container_width=True, height=316)
    left, right, spacer = st.columns([1, 1, 1.4])
    with left:
        st.download_button(
            "下载 result.json",
            data=json.dumps(data["results"], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="result.json",
            mime="application/json",
            type="primary",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "下载 result.txt",
            data=data["result_text"].encode("utf-8"),
            file_name="result.txt",
            mime="text/plain",
            use_container_width=True,
        )
    st.caption("下载内容来自匿名化缓存副本；原始 outputs 未被改动。")


def main() -> None:
    data = load_demo()
    st.sidebar.markdown("## AI 科研文献智能抽取助手")
    st.sidebar.caption("证据优先的科研抽取工作台")
    requested_page = query_value("page", "task")
    if requested_page not in PAGES:
        requested_page = "task"
    labels = list(PAGES.values())
    selected_label = st.sidebar.radio("工作区", labels, index=list(PAGES).index(requested_page), label_visibility="collapsed")
    page = next(key for key, label in PAGES.items() if label == selected_label)
    if page != requested_page:
        st.query_params["page"] = page
        st.rerun()
    st.sidebar.divider()
    st.sidebar.success("演示模式 · 缓存数据已就绪\n\n无需 API Key · 无需等待")
    st.sidebar.caption("公开包仅含匿名化展示层\n\n不包含科研主流程")

    renderers = {
        "task": lambda: render_task(data),
        "pipeline": lambda: render_pipeline(data),
        "samples": lambda: render_samples(data),
        "results": lambda: render_results(data),
        "evidence": lambda: render_evidence(data),
        "export": lambda: render_export(data),
    }
    renderers[page]()


if __name__ == "__main__":
    main()