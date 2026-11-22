/* 演示页的全部逻辑：一个 form 提交 → 调 /ask → 渲染答案和出处卡。
   没有框架，是因为这张页只需要做三件事（提问、显示答案、显示出处），
   引入构建工具反而让"clone 下来就能跑"这件事变难。 */

const $ = (sel) => document.querySelector(sel);

const form = $("#ask-form");
const input = $("#question");
const cohortSel = $("#cohort");
const submitBtn = $("#submit");
const panel = $("#answer-panel");
const sources = $("#sources");
const sourceList = $("#source-list");
const sourcesNote = $("#sources-note");
const empty = $("#empty");

const REASON_TEXT = {
  ok: "回答已通过引用校验",
  no_evidence: "检索到的片段相关度太低",
  citation_failed: "回答里引用的内容在原文里找不到",
  no_citation: "模型回答了，但没有标出处",
  model_abstained: "模型认为语料里没有依据",
  unindexed: "这一页的表格在 PDF 里是画出来的，读不到文字",
  api_error: "调用大模型接口失败",
};

const ABSTAIN_BOX = {
  unindexed: {
    icon: "📄",
    title: "这一页的表格读不出来",
    text: (d) => `命中的那几页，表格是用矢量线条画在 PDF 上的，文字层里只有零散数字，
      没有行列关系。最高相关度 ${d.top_score.toFixed(3)}。`,
  },
  model_abstained: {
    icon: "🔍",
    title: "模型判断资料里没有依据",
    text: () => `检索到的片段相关度不低，但模型读完认为答不出这个问题。`,
  },
  citation_failed: {
    icon: "🔍",
    title: "回答里的说法在原文中核对不上",
    text: (d) => `检索到的片段相关度是 ${d.top_score.toFixed(3)}，但模型写出的内容里有
      在原文找不到依据的部分，按规则弃用。`,
  },
  no_citation: {
    icon: "🔍",
    title: "回答没有标出处",
    text: () => `模型给了一段回答，但没有按要求标明出自哪一段，按规则弃用。`,
  },
  no_evidence: {
    icon: "🔍",
    title: "这个问题在培养方案里查不到可靠依据",
    text: (d) => `最高相关度 ${d.top_score.toFixed(3)}，低于拒答阈值 ${d.threshold}。`,
  },
  default: {
    icon: "🔍",
    title: "没有找到能支撑回答的原文",
    text: (d) => `最高相关度 ${d.top_score.toFixed(3)}。`,
  },
};

let busy = false;
let modelWarm = false;      // 后台是否已把向量/重排模型加载进内存（影响第一次提问的等待）

/* ---------- 把 [1][2] 变成可点的角标 ---------- */
function renderAnswer(text, used) {
  const frag = document.createElement("div");
  text.split(/\n{2,}/).forEach((para) => {
    const p = document.createElement("p");
    // 按 [n] 切开，普通文字直接进，引用编号做成角标
    para.split(/(\[\d{1,2}\])/).forEach((piece) => {
      const m = piece.match(/^\[(\d{1,2})\]$/);
      if (m) {
        const n = Number(m[1]);
        const span = document.createElement("span");
        span.className = "cite";
        span.textContent = n;
        span.title = `第 ${n} 段出处`;
        span.onclick = () => focusCard(n);
        p.appendChild(span);
      } else if (piece) {
        p.appendChild(document.createTextNode(piece));
      }
    });
    frag.appendChild(p);
  });
  return frag;
}

function focusCard(n) {
  const card = document.querySelector(`.card[data-n="${n}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("hot");
  setTimeout(() => card.classList.remove("hot"), 1600);
}

/* ---------- 出处卡 ---------- */
function renderSources(chunks, used) {
  sourceList.innerHTML = "";
  if (!chunks || !chunks.length) {
    sources.hidden = true;
    return;
  }
  sources.hidden = false;
  const cited = new Set(used || []);
  sourcesNote.textContent = cited.size
    ? `（答案里引用了 ${[...cited].sort((a, b) => a - b).join("、")} 号）`
    : "（模型没有引用其中任何一段）";

  chunks.forEach((c) => {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.n = c.n;

    const head = document.createElement("div");
    head.className = "card-head";
    head.innerHTML = `<span class="card-n">${c.n}</span>`;

    const path = document.createElement("div");
    path.className = "card-path";
    path.textContent = [c.title, c.path].filter(Boolean).join(" · ");
    head.appendChild(path);

    const sub = document.createElement("div");
    sub.className = "card-sub";
    sub.textContent = `相关度 ${c.score.toFixed(3)}`;
    head.appendChild(sub);
    card.appendChild(head);

    const bar = document.createElement("div");
    bar.className = "score";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(2, Math.min(100, c.score * 100))}%`;
    bar.appendChild(fill);
    card.appendChild(bar);

    const text = document.createElement("p");
    text.className = "card-text";
    text.textContent = c.text;
    card.appendChild(text);

    const more = document.createElement("button");
    more.className = "card-more";
    more.type = "button";
    more.textContent = "展开原文";
    more.onclick = () => {
      card.classList.toggle("open");
      more.textContent = card.classList.contains("open") ? "收起" : "展开原文";
    };
    card.appendChild(more);

    sourceList.appendChild(card);
  });
}

/* ---------- 主流程 ---------- */
async function ask(question, cohort) {
  if (busy || !question.trim()) return;
  busy = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "检索中";
  empty.hidden = true;
  sources.hidden = true;
  panel.hidden = false;
  panel.innerHTML = `<div class="skeleton"><i></i><i></i><i></i></div>
    <p style="margin:14px 0 0;font-size:13px;color:#7b8598">
      ${modelWarm
        ? "正在用关键词和向量各找 20 段、再让重排模型逐段读一遍，通常 2–4 秒…"
        : "模型还在加载（首次启动需要 20–30 秒），这一次会慢一些，之后每次 2–4 秒…"}</p>`;

  const started = performance.now();
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, cohort }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResult(data, Math.round(performance.now() - started));
  } catch (err) {
    panel.innerHTML = "";
    const box = document.createElement("div");
    box.className = "abstain";
    box.innerHTML = `<div class="icon">⚠️</div><div><h3>请求失败</h3>
      <p>服务没起来？先在项目根目录跑 <code>python api.py</code>，再刷新本页。
      错误信息：${String(err.message || err)}</p></div>`;
    panel.appendChild(box);
  } finally {
    busy = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "提问";
  }
}

function renderResult(data, ms) {
  panel.innerHTML = "";

  const head = document.createElement("div");
  head.className = "panel-head";
  const badge = document.createElement("span");
  if (data.abstained) {
    badge.className = "badge warn";
    badge.textContent = "拒答";
  } else {
    badge.className = "badge ok";
    badge.textContent = "已作答";
  }
  head.appendChild(badge);

  const reason = document.createElement("span");
  reason.textContent = REASON_TEXT[data.reason] || data.reason;
  head.appendChild(reason);
  head.appendChild(Object.assign(document.createElement("span"), { className: "spacer" }));
  const time = document.createElement("span");
  time.textContent = `${data.latency_ms.total} ms`;
  head.appendChild(time);
  panel.appendChild(head);

  if (data.abstained) {
    const box = document.createElement("div");
    box.className = "abstain";
    const absorb = ABSTAIN_BOX[data.reason] || ABSTAIN_BOX.default;
    box.innerHTML = `<div class="icon">${absorb.icon}</div><div>
      <h3>${absorb.title}</h3><p>${absorb.text(data)}</p></div>`;
    panel.appendChild(box);
    if (data.answer && !data.fallback) {  // 拒答但模型给了一段解释（例如"这页表格读不到"）
      const note = document.createElement("div");
      note.className = "answer abstain-answer";
      note.appendChild(renderAnswer(data.answer, data.used));
      panel.appendChild(note);
    }
  } else {
    const answer = document.createElement("div");
    answer.className = "answer";
    answer.appendChild(renderAnswer(data.answer, data.used));
    panel.appendChild(answer);
  }

  const meta = document.createElement("div");
  meta.className = "meta-line";
  meta.innerHTML = [
    `<span>最高相关度 <b>${data.top_score.toFixed(3)}</b></span>`,
    data.abstained ? "" : `<span>引用校验 <b>${data.citation_ok ? "通过" : "未通过"}</b></span>`,
    `<span>检索耗时 <b>${data.latency_ms.search} ms</b></span>`,
    `<span>总耗时 <b>${data.latency_ms.total} ms</b></span>`,
    data.cached ? "<span>（本次命中缓存）</span>" : "",
  ].join("");
  panel.appendChild(meta);

  renderSources(data.chunks, data.used);
}

/* ---------- 事件 ---------- */
form.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(input.value, cohortSel.value);
});

$("#samples").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-q]");
  if (!btn) return;
  input.value = btn.dataset.q;
  cohortSel.value = btn.dataset.c || "";
  ask(input.value, cohortSel.value);
});

fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    if (h.chunks) $("#stat-chunks").textContent = h.chunks;
    if (!h.llm_ready) {
      const tip = document.createElement("p");
      tip.className = "tagline";
      tip.textContent = "提示：服务没读到 DEEPSEEK_API_KEY，只能检索不能生成答案。";
      document.querySelector(".hero .wrap").appendChild(tip);
    }
    watchModel(h);
  })
  .catch(() => {});

/* 模型是在后台线程里加载的：页面先开，加载完再放人提问。
   不这样做的话，第一个问题会把 20~30 秒的模型加载时间算进"检索耗时"，
   看起来像检索慢，其实是还没进内存。 */
function watchModel(h) {
  const tip = document.createElement("p");
  tip.className = "tagline";
  tip.id = "boot-status";
  tip.style.color = "#b8860b";
  document.querySelector(".hero .wrap").appendChild(tip);

  const show = (state) => {
    modelWarm = state;
    tip.textContent = state ? "就绪，可以提问了" : "正在加载检索与重排模型（首次启动约 20–30 秒）…";
    if (state) setTimeout(() => { tip.remove(); }, 4000);
  };
  if (h.warm) { show(true); return; }
  show(false);

  const timer = setInterval(async () => {
    try {
      const r = await fetch("/health");
      const d = await r.json();
      if (d.warm) { clearInterval(timer); show(true); }
    } catch (e) { clearInterval(timer); }
  }, 1500);
}

/* 支持带参数直接打开，方便把一个问题做成链接（截图、演示用）：
   http://127.0.0.1:8000/?q=2026级数字经济专业毕业总学分&cohort=2026级 */
const params = new URLSearchParams(location.search);
if (params.get("q")) {
  input.value = params.get("q");
  cohortSel.value = params.get("cohort") || "";
  ask(input.value, cohortSel.value);
}
