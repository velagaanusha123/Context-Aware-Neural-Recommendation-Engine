/* ─────────────────────────────────────────────
   script.js — H&M Recommendation Engine UI
   ───────────────────────────────────────────── */

const API = "http://127.0.0.1:8000";

const GARMENT_ICONS = {
  "Jersey Basic": "👕",
  "Knitwear": "🧶",
  "Dresses Ladies": "👗",
  "Trousers": "👖",
  "Shorts": "🩳",
  "Skirts": "🩴",
  "Outdoor": "🧥",
  "Accessories": "👜",
  "Socks & Tights": "🧦",
  "Shoes": "👟",
  "Swimwear": "🩱",
  "Underwear": "🩲",
  "Nightwear": "🌙",
  "Bags": "👜",
  "Hats & Scarves": "🧣",
};

const MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

let cataloguePage = 0;
const PAGE_SIZE = 20;

window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadUsers();
  loadCatalogue();

  // Sync range label
  const range = document.getElementById("topk-range");
  range.addEventListener("input", () => {
    document.getElementById("topk-val").textContent = range.value;
  });

  // Update everything when user changes
  document.getElementById("user-select").addEventListener("change", () => {
    const userId = document.getElementById("user-select").value;

    // 1. Refresh Recommendations if visible
    if (!document.getElementById("panel-recs").classList.contains("hidden")) {
      fetchRecommendations();
    }
    // 2. Refresh Analytics if visible
    if (!document.getElementById("panel-analytics").classList.contains("hidden")) {
      loadAnalytics();
    }

    // 3. Update the Hero Title to show the current user
    if (userId) {
      document.querySelector(".hero-text h2").textContent = `Fashion Picks for User #${userId}`;
    }
  });
});

// ─── Health check ──────────────────────────────────────────────────────────

async function checkHealth() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const r = await fetch(`${API}/`);
    if (r.ok) {
      dot.classList.add("online");
      text.textContent = "API Online";
    } else throw new Error();
  } catch {
    dot.classList.add("offline");
    text.textContent = "API Offline";
  }
}

// ─── Load users ────────────────────────────────────────────────────────────

async function loadUsers() {
  const sel = document.getElementById("user-select");
  try {
    const r = await fetch(`${API}/users/sample?n=200`);
    const data = await r.json();
    sel.innerHTML = data.user_ids
      .map(u => `<option value="${u}">User #${u}</option>`)
      .join("");
  } catch {
    sel.innerHTML = `<option value="">⚠ Could not load users</option>`;
  }
}

// ─── Recommendations ───────────────────────────────────────────────────────

async function fetchRecommendations() {
  const userId = parseInt(document.getElementById("user-select").value);
  const topK = parseInt(document.getElementById("topk-range").value);
  const monthRaw = document.getElementById("month-select").value;
  const month = monthRaw ? parseInt(monthRaw) : null;

  if (isNaN(userId)) return;

  const btn = document.getElementById("recommend-btn");
  const grid = document.getElementById("rec-grid");
  const loader = document.getElementById("rec-loader");

  btn.disabled = true;
  grid.innerHTML = "";
  loader.classList.remove("hidden");

  const t0 = performance.now();

  try {
    const body = { user_idx: userId, top_k: topK };
    if (month) body.month_filter = month;

    const r = await fetch(`${API}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Unknown error");
    }

    const data = await r.json();
    const latency = Math.round(performance.now() - t0);

    loader.classList.add("hidden");

    // Update stats
    document.getElementById("stat-user").textContent = `#${userId}`;
    document.getElementById("stat-count").textContent = data.recommendations.length;
    document.getElementById("stat-month").textContent = month ? MONTH_NAMES[month] : "All";
    document.getElementById("stat-latency").textContent = `${latency} ms`;

    if (!data.recommendations.length) {
      grid.innerHTML = `<div class="empty-state"><div class="empty-icon">😔</div>
        <p>No recommendations found for this user / filter combo.</p></div>`;
      return;
    }

    // Find max score for normalising bar widths
    const maxScore = Math.max(...data.recommendations.map(r => r.score || 0)) || 1;

    grid.innerHTML = data.recommendations
      .map((item, i) => renderItemCard(item, i, maxScore, true))
      .join("");

  } catch (err) {
    loader.classList.add("hidden");
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div>
      <p>${err.message}</p></div>`;
  } finally {
    btn.disabled = false;
  }
}

// ─── Catalogue ─────────────────────────────────────────────────────────────

async function loadCatalogue() {
  const grid = document.getElementById("cat-grid");
  grid.innerHTML = `<div class="empty-state"><div class="spinner"></div><p>Loading…</p></div>`;

  try {
    const offset = cataloguePage * PAGE_SIZE;
    const r = await fetch(`${API}/items?limit=${PAGE_SIZE}&offset=${offset}`);
    const data = await r.json();

    if (!data.items.length) {
      grid.innerHTML = `<div class="empty-state"><div class="empty-icon">📦</div><p>No items.</p></div>`;
      return;
    }

    grid.innerHTML = data.items
      .map((item, i) => renderItemCard(item, i, null, false))
      .join("");

    document.getElementById("cat-page").textContent = `Page ${cataloguePage + 1}`;
    document.getElementById("cat-prev").disabled = cataloguePage === 0;

  } catch (err) {
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div>
      <p>Could not load catalogue: ${err.message}</p></div>`;
  }
}

function changePage(dir) {
  cataloguePage = Math.max(0, cataloguePage + dir);
  loadCatalogue();
}

// ─── Render helpers ────────────────────────────────────────────────────────

function renderItemCard(item, index, maxScore, showScore) {
  const icon = GARMENT_ICONS[item.garment_group] || "🏷";
  const delay = `animation-delay:${index * 0.04}s`;

  // Image logic
  const artId = String(item.article_id).padStart(10, '0');
  const folder = artId.substring(0, 3);
  // We use a relative path. If the image doesn't exist, we fallback to icon via CSS/JS
  const imgPath = `../assets/images/${folder}/${artId}.jpg`;

  const imgHtml = `<img src="${imgPath}" class="item-card-img" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">`;
  const iconHtml = `
  <img 
    src="https://via.placeholder.com/300x400/1a1a27/ffffff?text=Fashion+Item"
    class="item-card-img"
    style="display:none;"
  >
`;

  const scoreHtml = showScore && item.score != null ? `
    <div class="score-bar-wrap">
      <div class="score-label">
        <span>Relevance</span>
        <span>${(item.score / maxScore * 100).toFixed(0)}%</span>
      </div>
      <div class="score-bar">
        <div class="score-fill" style="width:${(item.score / maxScore * 100).toFixed(1)}%"></div>
      </div>
    </div>` : "";

  const popHtml = item.popularity > 0
    ? `<div class="pop-badge">⭐ ${item.popularity.toLocaleString()} purchases</div>` : "";

  return `
    <div class="item-card" style="${delay}">
      ${imgHtml}
      ${iconHtml}
      <div class="item-card-name">${escHtml(item.prod_name)}</div>
      <div class="item-card-type">${escHtml(item.product_type || "")} · ${escHtml(item.product_group || "")}</div>
      <div class="item-card-tags">
        ${item.colour ? `<span class="tag accent2">${escHtml(item.colour)}</span>` : ""}
        ${item.section ? `<span class="tag">${escHtml(item.section)}</span>` : ""}
        ${item.garment_group ? `<span class="tag accent">${escHtml(item.garment_group)}</span>` : ""}
      </div>
      ${scoreHtml}
      ${popHtml}
    </div>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ─── Tab switching ─────────────────────────────────────────────────────────

function switchTab(name) {
  document.getElementById("panel-recs").classList.toggle("hidden", name !== "recs");
  document.getElementById("panel-browse").classList.toggle("hidden", name !== "browse");
  document.getElementById("panel-analytics").classList.toggle("hidden", name !== "analytics");
  document.getElementById("panel-train").classList.toggle("hidden", name !== "train");

  document.getElementById("tab-recs").classList.toggle("active", name === "recs");
  document.getElementById("tab-browse").classList.toggle("active", name === "browse");
  document.getElementById("tab-analytics").classList.toggle("active", name === "analytics");
  document.getElementById("tab-train").classList.toggle("active", name === "train");

  if (name === 'analytics') loadAnalytics();
}

// ─── Analytics ─────────────────────────────────────────────────────────────

async function loadAnalytics() {
  try {
    // Load stats
    const rStats = await fetch(`${API}/analytics/stats`);
    const stats = await rStats.json();
    document.getElementById("count-users").textContent = stats.users.toLocaleString();
    document.getElementById("count-items").textContent = stats.items.toLocaleString();
    document.getElementById("count-interactions").textContent = stats.interactions.toLocaleString();

    // Load seasonal trends
    const rTrends = await fetch(`${API}/analytics/seasonal`);
    const trends = await rTrends.json();

    const traceData = {};
    trends.forEach(d => {
      if (!traceData[d.category]) traceData[d.category] = { x: [], y: [], name: d.category, type: 'scatter', mode: 'lines+markers' };
      traceData[d.category].x.push(d.month);
      traceData[d.category].y.push(d.count);
    });

    const layout = {
      title: 'Top Fashion Categories by Month',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#8888aa' },
      xaxis: { gridcolor: '#2a2a40' },
      yaxis: { gridcolor: '#2a2a40' },
      margin: { t: 50, b: 50, l: 50, r: 20 }
    };

    Plotly.newPlot('chart-seasonal', Object.values(traceData), layout);

    // Load user-specific activity
    loadUserAnalytics();

  } catch (err) {
    console.error("Analytics load failed", err);
  }
}

// ─── Training ──────────────────────────────────────────────────────────────

async function triggerTraining() {
  const btn = document.getElementById("train-btn");
  const progressContainer = document.getElementById("train-progress-container");
  const progressFill = document.getElementById("train-progress-fill");
  const statusText = document.getElementById("train-status");

  btn.disabled = true;
  progressContainer.classList.remove("hidden");
  progressFill.style.width = "0%";
  statusText.textContent = "Connecting to server...";

  try {
    // Simulate progress for UI
    let progress = 0;
    const interval = setInterval(() => {
      if (progress < 90) {
        progress += Math.random() * 5;
        progressFill.style.width = `${progress}%`;
        statusText.textContent = `Processing dataset... (${Math.round(progress)}%)`;
      }
    }, 800);

    const r = await fetch(`${API}/train`, { method: 'POST' });
    clearInterval(interval);

    if (r.ok) {
      progressFill.style.width = "100%";
      statusText.textContent = "Neural model trained and loaded!";
      alert("Success! The Neural model has been updated.");
    } else {
      throw new Error("Training failed on server.");
    }
  } catch (err) {
    statusText.textContent = "Error: " + err.message;
    progressFill.style.backgroundColor = "#f87171";
  } finally {
    btn.disabled = false;
  }
}

async function loadUserAnalytics() {
  const userId = parseInt(document.getElementById("user-select").value);
  if (isNaN(userId)) return;

  try {
    const r = await fetch(`${API}/analytics/user/${userId}`);
    const data = await r.json();

    const trace = {
      x: data.map(d => d.month),
      y: data.map(d => d.count),
      type: 'bar',
      marker: { color: '#ff6fb0' },
      name: `User #${userId} History`
    };

    const layout = {
      title: `User #${userId}: Purchase History`,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#8888aa' },
      xaxis: { gridcolor: '#2a2a40' },
      yaxis: { gridcolor: '#2a2a40' },
      margin: { t: 50, b: 50, l: 50, r: 20 }
    };

    Plotly.newPlot('chart-user', [trace], layout);
  } catch (err) {
    console.error("User analytics failed", err);
  }
}
