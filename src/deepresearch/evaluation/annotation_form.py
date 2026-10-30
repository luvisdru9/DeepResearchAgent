"""Render dependency-free browser forms for human benchmark annotation.

The form deliberately embeds only one annotator packet.  It never calls a
server and does not preselect labels, which keeps independent human judgments
separate while making the frozen JSON packets easier to complete.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from deepresearch.evaluation.annotations import HumanAnnotationBundle


def render_annotation_form(bundle: HumanAnnotationBundle, *, source_name: str) -> str:
    """Return a self-contained HTML form for one annotation bundle."""

    # Escape script-closing characters even though packets currently contain
    # trusted benchmark text; future web-derived claims may contain arbitrary HTML.
    payload = bundle.model_dump(mode="json")
    embedded_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = "独立标注表" if bundle.role == "annotator" else "分歧裁决表"
    role_warning = (
        "请独立完成，不要查看另一位标注者的答案。"
        if bundle.role == "annotator"
        else "仅在 A/B 独立标注完成并确认分歧后填写。"
    )
    return _HTML_TEMPLATE.replace("{{TITLE}}", escape(title)).replace(
        "{{ROLE_WARNING}}", escape(role_warning)
    ).replace("{{SOURCE_NAME}}", escape(source_name)).replace(
        "{{PACKET_JSON}}", embedded_json
    )


def write_annotation_form(
    bundle: HumanAnnotationBundle,
    destination: str | Path,
    *,
    source_name: str,
) -> Path:
    """Write one self-contained form without changing its source packet."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_annotation_form(bundle, source_name=source_name), encoding="utf-8"
    )
    return path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{TITLE}}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #dce2ea;
      --soft: #f5f7fa;
      --panel: #ffffff;
      --brand: #2855a6;
      --brand-soft: #eaf0fb;
      --ok: #18794e;
      --warn: #a15c00;
      --error: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #eef2f6;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      line-height: 1.55;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 14px 24px;
      color: white;
      background: #172c52;
      box-shadow: 0 2px 8px #17203333;
    }
    header h1 { margin: 0; font-size: 1.25rem; }
    header p { margin: 3px 0 0; color: #dbe6f8; font-size: .9rem; }
    .layout {
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(0, 900px);
      gap: 20px;
      max-width: 1240px;
      margin: 20px auto;
      padding: 0 20px 80px;
      align-items: start;
    }
    aside {
      position: sticky;
      top: 92px;
      display: grid;
      gap: 14px;
    }
    .panel, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 2px 8px #1720330d;
    }
    .panel { padding: 16px; }
    .panel h2, .panel h3 { margin: 0 0 10px; }
    .panel h2 { font-size: 1.05rem; }
    .panel h3 { font-size: .95rem; }
    .small { color: var(--muted); font-size: .84rem; }
    .warning {
      padding: 10px 12px;
      color: #6f4100;
      background: #fff6df;
      border-left: 4px solid #e6a700;
      border-radius: 6px;
    }
    label.field-label { display: block; margin-bottom: 5px; font-weight: 650; }
    input[type="text"], textarea {
      width: 100%;
      padding: 9px 10px;
      color: var(--ink);
      background: white;
      border: 1px solid #b8c2d0;
      border-radius: 7px;
      font: inherit;
    }
    textarea { min-height: 72px; resize: vertical; }
    input:focus, textarea:focus {
      outline: 3px solid #82a9ee55;
      border-color: var(--brand);
    }
    progress { width: 100%; height: 14px; accent-color: var(--ok); }
    .progress-text { margin-top: 5px; font-size: .9rem; font-weight: 650; }
    .actions { display: grid; gap: 8px; }
    button, .file-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 8px 12px;
      border: 1px solid #9eabba;
      border-radius: 8px;
      color: var(--ink);
      background: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover, .file-button:hover { background: var(--soft); }
    button.primary { color: white; background: var(--brand); border-color: var(--brand); }
    button.primary:hover { background: #1f478f; }
    .file-button input { display: none; }
    .sample-nav { display: grid; gap: 6px; }
    .sample-nav button { justify-content: space-between; font-weight: 500; }
    .sample-nav button.active { color: var(--brand); background: var(--brand-soft); border-color: #9db7e6; }
    main { min-width: 0; }
    .sample { display: none; }
    .sample.active { display: block; }
    .sample-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 14px;
    }
    .sample-title h2 { margin: 0; }
    .section-title { margin: 26px 0 10px; }
    .section-help {
      padding: 10px 12px;
      margin-bottom: 12px;
      color: #28436b;
      background: #edf4ff;
      border-radius: 8px;
      font-size: .9rem;
    }
    .card { padding: 18px; margin-bottom: 14px; scroll-margin-top: 105px; }
    .card.incomplete { border-color: #e6a700; box-shadow: 0 0 0 2px #e6a70022; }
    .card.complete { border-left: 5px solid var(--ok); }
    .card-head { display: flex; justify-content: space-between; gap: 12px; }
    .card-id { color: var(--muted); font: .82rem ui-monospace, monospace; }
    .badge { padding: 2px 8px; border-radius: 999px; background: var(--soft); font-size: .78rem; }
    .claim-text { margin: 12px 0; font-size: 1.02rem; font-weight: 650; }
    .metadata { color: var(--muted); font-size: .84rem; }
    details {
      margin: 10px 0;
      padding: 8px 10px;
      background: var(--soft);
      border-radius: 7px;
    }
    details summary { color: #28436b; font-weight: 650; cursor: pointer; }
    blockquote { margin: 9px 0; padding-left: 12px; border-left: 3px solid #9db7e6; }
    .label-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; margin: 12px 0; }
    .choice {
      display: block;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 7px;
      cursor: pointer;
    }
    .choice:hover { background: var(--soft); }
    .choice:has(input:checked) { background: var(--brand-soft); border-color: #789bd7; }
    .choice input { margin-right: 6px; }
    .choice small { display: block; margin-left: 23px; color: var(--muted); }
    .claim-links { display: grid; gap: 6px; margin: 10px 0 14px; }
    .claim-link { padding: 7px 9px; background: var(--soft); border-radius: 7px; }
    .claim-link input { margin-right: 7px; }
    .status-message { min-height: 24px; margin-top: 8px; font-size: .88rem; }
    .status-message.error { color: var(--error); }
    .status-message.ok { color: var(--ok); }
    .legend { margin: 0; padding-left: 18px; font-size: .84rem; }
    .legend li { margin: 5px 0; }
    .footer-note { margin-top: 22px; color: var(--muted); text-align: center; font-size: .82rem; }
    @media (max-width: 800px) {
      .layout { grid-template-columns: 1fr; padding: 0 12px 60px; }
      aside { position: static; }
      .label-group { grid-template-columns: 1fr; }
      header { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <h1>{{TITLE}}</h1>
    <p>来源：{{SOURCE_NAME}} · 数据只在当前浏览器页面处理，不会上传。</p>
  </header>
  <div class="layout">
    <aside>
      <section class="panel">
        <h2>填写者</h2>
        <p class="warning">{{ROLE_WARNING}}</p>
        <label class="field-label" for="annotator-id">匿名标注者 ID</label>
        <input id="annotator-id" type="text" autocomplete="off" placeholder="例如 annotator_a">
        <p class="small">不要填写姓名、邮箱等个人信息；不能保留 <code>_pending</code>。</p>
      </section>
      <section class="panel">
        <h2>总体进度</h2>
        <progress id="progress" value="0" max="1"></progress>
        <div id="progress-text" class="progress-text">0 / 0</div>
        <div id="sample-nav" class="sample-nav"></div>
      </section>
      <section class="panel actions">
        <h2>保存与检查</h2>
        <label class="file-button">导入已保存草稿<input id="import-file" type="file" accept="application/json,.json"></label>
        <button id="export-draft" type="button">导出当前草稿 JSON</button>
        <button id="find-missing" type="button">定位下一处未填写</button>
        <button id="export-complete" class="primary" type="button">检查并导出完成版</button>
        <div id="status-message" class="status-message" aria-live="polite"></div>
      </section>
      <section class="panel">
        <h3>判断原则</h3>
        <ul class="legend">
          <li>只使用当前表单提供的 claim 和 Evidence。</li>
          <li>不要因为来源看起来权威就判为支持。</li>
          <li>多事实 claim 必须全部得到支持才算 supported。</li>
          <li>每一项都填写简短、可复核的理由。</li>
        </ul>
      </section>
    </aside>
    <main id="samples"></main>
  </div>
  <script id="packet-data" type="application/json">{{PACKET_JSON}}</script>
  <script>
  (() => {
    "use strict";
    const original = JSON.parse(document.getElementById("packet-data").textContent);
    let data = structuredClone(original);
    let activeSample = 0;

    const claimLabels = [
      ["supported", "全部重要事实均被证据直接支持"],
      ["partially_supported", "只支持部分内容，或表述强于证据"],
      ["unsupported", "引用有效，但证据不能推出该陈述"],
      ["contradicted", "证据与陈述明确冲突"],
      ["miscited", "引用编号无效或指向错误证据"],
      ["uncited", "需要证据，但没有引用标记"]
    ];
    const goldLabels = [
      ["covered", "报告完整表达了标准要点"],
      ["partially_covered", "只覆盖标准要点的一部分"],
      ["not_covered", "报告没有覆盖该要点"],
      ["contradicted", "报告给出了相反结论"]
    ];

    const samplesRoot = document.getElementById("samples");
    const navRoot = document.getElementById("sample-nav");
    const statusRoot = document.getElementById("status-message");
    const annotatorInput = document.getElementById("annotator-id");
    annotatorInput.value = data.annotator_id.replace(/_pending$/, "");
    annotatorInput.addEventListener("input", () => {
      data.annotator_id = annotatorInput.value.trim();
      data.samples.forEach(sample => { sample.annotator_id = data.annotator_id; });
    });

    function element(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function render() {
      samplesRoot.replaceChildren();
      navRoot.replaceChildren();
      data.samples.forEach((sample, sampleIndex) => {
        const nav = element("button", sampleIndex === activeSample ? "active" : "");
        nav.type = "button";
        nav.append(element("span", "", sample.sample_id));
        nav.append(element("span", "badge", sampleProgressText(sample)));
        nav.addEventListener("click", () => showSample(sampleIndex));
        navRoot.append(nav);

        const section = element("section", "sample" + (sampleIndex === activeSample ? " active" : ""));
        section.dataset.sampleIndex = String(sampleIndex);
        const title = element("div", "sample-title");
        const heading = element("h2", "", `样本 ${sampleIndex + 1}：${sample.sample_id}`);
        title.append(heading, element("span", "badge", `run ${sample.run_id}`));
        section.append(title);
        section.append(sectionHelp(
          "A. 报告陈述与引用证据",
          "逐条比较报告 claim 与下方 Evidence；不要使用外部知识补足证据。"
        ));
        sample.report_claims.forEach((item, itemIndex) => {
          section.append(reportClaimCard(sample, sampleIndex, item, itemIndex));
        });
        section.append(sectionHelp(
          "B. 标准要点覆盖",
          "判断整份报告是否覆盖 gold claim，并勾选实际参与覆盖的 report claim。"
        ));
        sample.gold_claims.forEach((item, itemIndex) => {
          section.append(goldClaimCard(sample, sampleIndex, item, itemIndex));
        });
        section.append(element("p", "footer-note", "样本内全部项目完成后，导出程序会自动把 status 设为 complete。"));
        samplesRoot.append(section);
      });
      updateProgress();
    }

    function sectionHelp(titleText, helpText) {
      const fragment = document.createDocumentFragment();
      const heading = element("h3", "section-title", titleText);
      const help = element("div", "section-help", helpText);
      fragment.append(heading, help);
      return fragment;
    }

    function reportClaimCard(sample, sampleIndex, item, itemIndex) {
      const card = baseCard(item.claim_id, "report", sampleIndex, itemIndex);
      card.append(element("p", "claim-text", item.claim));
      card.append(element(
        "p", "metadata",
        `引用编号：${item.citation_numbers.length ? item.citation_numbers.join(", ") : "无"} · Evidence：${item.evidence_ids.length ? item.evidence_ids.join(", ") : "无"}`
      ));
      if (item.evidence.length) {
        item.evidence.forEach(evidence => {
          const details = document.createElement("details");
          details.open = item.evidence.length === 1;
          const summary = element("summary", "", `${evidence.evidence_id} · ${evidence.source_title}`);
          const source = element("p", "metadata", evidence.source_url);
          const quote = element("blockquote", "", evidence.quote || "（无 quote）");
          details.append(summary, source, quote);
          if (evidence.claim && evidence.claim !== evidence.quote) {
            details.append(element("p", "small", `Evidence 摘要：${evidence.claim}`));
          }
          card.append(details);
        });
      } else {
        card.append(element("p", "warning", "本条没有可供核对的 Evidence。请先检查是否属于 uncited 或 miscited。"));
      }
      card.append(labelChoices(claimLabels, item.label, value => {
        item.label = value;
        refreshCard(card, item);
      }, `report-${sampleIndex}-${itemIndex}`));
      card.append(reasonField(item, card));
      refreshCard(card, item);
      return card;
    }

    function goldClaimCard(sample, sampleIndex, item, itemIndex) {
      const card = baseCard(item.gold_claim_id, "gold", sampleIndex, itemIndex);
      card.append(element("p", "claim-text", item.gold_claim));
      card.append(element("p", "metadata", `权重：${item.weight} · 必需：${item.required ? "是" : "否"}`));
      const links = element("div", "claim-links");
      links.append(element("strong", "", "勾选覆盖该要点的报告 claim："));
      sample.report_claims.forEach(reportClaim => {
        const label = element("label", "claim-link");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = item.report_claim_ids.includes(reportClaim.claim_id);
        checkbox.addEventListener("change", () => {
          if (checkbox.checked && !item.report_claim_ids.includes(reportClaim.claim_id)) {
            item.report_claim_ids.push(reportClaim.claim_id);
          } else if (!checkbox.checked) {
            item.report_claim_ids = item.report_claim_ids.filter(id => id !== reportClaim.claim_id);
          }
          refreshCard(card, item);
        });
        const text = element("span", "", `${reportClaim.claim_id} — ${reportClaim.claim}`);
        label.append(checkbox, text);
        links.append(label);
      });
      card.append(links);
      card.append(labelChoices(goldLabels, item.label, value => {
        item.label = value;
        if (value === "not_covered") item.report_claim_ids = [];
        render();
      }, `gold-${sampleIndex}-${itemIndex}`));
      card.append(reasonField(item, card));
      refreshCard(card, item);
      return card;
    }

    function baseCard(id, kind, sampleIndex, itemIndex) {
      const card = element("article", "card");
      card.dataset.kind = kind;
      card.dataset.sampleIndex = String(sampleIndex);
      card.dataset.itemIndex = String(itemIndex);
      const head = element("div", "card-head");
      head.append(element("span", "card-id", id), element("span", "badge", kind === "report" ? "证据支持" : "要点覆盖"));
      card.append(head);
      return card;
    }

    function labelChoices(options, selected, onChange, name) {
      const group = element("div", "label-group");
      options.forEach(([value, description]) => {
        const label = element("label", "choice");
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = name;
        radio.value = value;
        radio.checked = selected === value;
        radio.addEventListener("change", () => onChange(value));
        label.append(radio, document.createTextNode(value), element("small", "", description));
        group.append(label);
      });
      return group;
    }

    function reasonField(item, card) {
      const wrapper = element("div", "");
      const label = element("label", "field-label", "判断理由（必填）");
      const textarea = document.createElement("textarea");
      textarea.placeholder = "指出证据具体支持、缺失或冲突的部分；不要只写“看起来正确”。";
      textarea.value = item.reason || "";
      textarea.addEventListener("input", () => {
        item.reason = textarea.value;
        refreshCard(card, item);
      });
      wrapper.append(label, textarea);
      return wrapper;
    }

    function itemComplete(item) {
      return Boolean(item.label && item.reason && item.reason.trim());
    }

    function refreshCard(card, item) {
      card.classList.toggle("complete", itemComplete(item));
      card.classList.toggle("incomplete", !itemComplete(item));
      updateProgress();
    }

    function allItems(sample) {
      return [...sample.report_claims, ...sample.gold_claims];
    }

    function sampleProgressText(sample) {
      const items = allItems(sample);
      return `${items.filter(itemComplete).length}/${items.length}`;
    }

    function updateProgress() {
      const items = data.samples.flatMap(allItems);
      const done = items.filter(itemComplete).length;
      const progress = document.getElementById("progress");
      progress.max = Math.max(items.length, 1);
      progress.value = done;
      document.getElementById("progress-text").textContent = `${done} / ${items.length} 项完成`;
      [...navRoot.children].forEach((button, index) => {
        const badge = button.querySelector(".badge");
        if (badge) badge.textContent = sampleProgressText(data.samples[index]);
      });
    }

    function showSample(index) {
      activeSample = index;
      document.querySelectorAll(".sample").forEach((node, nodeIndex) => node.classList.toggle("active", nodeIndex === index));
      [...navRoot.children].forEach((node, nodeIndex) => node.classList.toggle("active", nodeIndex === index));
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function firstMissing() {
      for (let sampleIndex = 0; sampleIndex < data.samples.length; sampleIndex += 1) {
        const sample = data.samples[sampleIndex];
        for (const [kind, items] of [["report", sample.report_claims], ["gold", sample.gold_claims]]) {
          const itemIndex = items.findIndex(item => !itemComplete(item));
          if (itemIndex >= 0) return { sampleIndex, kind, itemIndex };
        }
      }
      return null;
    }

    function locateMissing() {
      const missing = firstMissing();
      if (!missing) {
        setStatus("所有项目均已填写，可以导出完成版。", "ok");
        return;
      }
      showSample(missing.sampleIndex);
      const selector = `.card[data-kind="${missing.kind}"][data-sample-index="${missing.sampleIndex}"][data-item-index="${missing.itemIndex}"]`;
      setTimeout(() => document.querySelector(selector)?.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
      setStatus("已定位到下一处缺少标签或理由的项目。", "error");
    }

    function exportPayload(completed) {
      const annotatorId = annotatorInput.value.trim();
      if (!annotatorId || /_pending$/.test(annotatorId)) {
        throw new Error("请填写不以 _pending 结尾的匿名标注者 ID。");
      }
      if (completed && firstMissing()) {
        throw new Error("仍有项目缺少标签或理由，请先定位并补全。");
      }
      const output = structuredClone(data);
      output.annotator_id = annotatorId;
      output.samples.forEach(sample => {
        sample.annotator_id = annotatorId;
        sample.status = completed ? "complete" : "pending";
      });
      return output;
    }

    function download(payload, suffix) {
      const safeId = payload.annotator_id.replace(/[^A-Za-z0-9._-]+/g, "_");
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${safeId}_${suffix}.json`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function setStatus(message, kind) {
      statusRoot.textContent = message;
      statusRoot.className = `status-message ${kind || ""}`;
    }

    document.getElementById("find-missing").addEventListener("click", locateMissing);
    document.getElementById("export-draft").addEventListener("click", () => {
      try {
        download(exportPayload(false), "draft");
        setStatus("草稿已下载；浏览器不会自动写回仓库。", "ok");
      } catch (error) { setStatus(error.message, "error"); }
    });
    document.getElementById("export-complete").addEventListener("click", () => {
      try {
        download(exportPayload(true), "completed");
        setStatus("完成版已下载。请把 JSON 放回 annotations 目录并进行 schema 校验。", "ok");
      } catch (error) {
        setStatus(error.message, "error");
        locateMissing();
      }
    });
    document.getElementById("import-file").addEventListener("change", async event => {
      const file = event.target.files[0];
      if (!file) return;
      try {
        const incoming = JSON.parse(await file.text());
        if (incoming.role !== original.role || incoming.samples?.length !== original.samples.length) {
          throw new Error("草稿角色或样本数量与当前表单不一致。");
        }
        const originalKeys = original.samples.map(sample => `${sample.sample_id}:${sample.run_id}`).join("|");
        const incomingKeys = incoming.samples.map(sample => `${sample.sample_id}:${sample.run_id}`).join("|");
        if (originalKeys !== incomingKeys) throw new Error("草稿样本 ID/run ID 与当前表单不一致。");
        // Only merge human-editable fields. Claims, Evidence and stable IDs
        // always come from the frozen packet embedded in this form.
        const merged = structuredClone(original);
        const validClaimLabels = new Set(claimLabels.map(([value]) => value));
        const validGoldLabels = new Set(goldLabels.map(([value]) => value));
        merged.annotator_id = String(incoming.annotator_id || "").trim();
        merged.samples.forEach((sample, sampleIndex) => {
          const draftSample = incoming.samples[sampleIndex];
          const reportIds = sample.report_claims.map(item => item.claim_id);
          const draftReportIds = draftSample.report_claims?.map(item => item.claim_id) || [];
          const goldIds = sample.gold_claims.map(item => item.gold_claim_id);
          const draftGoldIds = draftSample.gold_claims?.map(item => item.gold_claim_id) || [];
          if (reportIds.join("|") !== draftReportIds.join("|")) {
            throw new Error(`样本 ${sample.sample_id} 的 report claim ID 与冻结包不一致。`);
          }
          if (goldIds.join("|") !== draftGoldIds.join("|")) {
            throw new Error(`样本 ${sample.sample_id} 的 gold claim ID 与冻结包不一致。`);
          }
          sample.annotator_id = merged.annotator_id;
          sample.report_claims.forEach((item, itemIndex) => {
            const draftItem = draftSample.report_claims[itemIndex];
            item.label = validClaimLabels.has(draftItem.label) ? draftItem.label : null;
            item.reason = typeof draftItem.reason === "string" ? draftItem.reason : "";
          });
          const allowedReportIds = new Set(reportIds);
          sample.gold_claims.forEach((item, itemIndex) => {
            const draftItem = draftSample.gold_claims[itemIndex];
            item.label = validGoldLabels.has(draftItem.label) ? draftItem.label : null;
            item.reason = typeof draftItem.reason === "string" ? draftItem.reason : "";
            item.report_claim_ids = Array.isArray(draftItem.report_claim_ids)
              ? [...new Set(draftItem.report_claim_ids.filter(id => allowedReportIds.has(id)))]
              : [];
            if (item.label === "not_covered") item.report_claim_ids = [];
          });
        });
        data = merged;
        annotatorInput.value = data.annotator_id.replace(/_pending$/, "");
        render();
        setStatus("草稿填写内容已载入；claim、Evidence 和稳定 ID 仍来自冻结表单。", "ok");
      } catch (error) { setStatus(`无法导入：${error.message}`, "error"); }
      event.target.value = "";
    });

    render();
  })();
  </script>
</body>
</html>
'''
