const { useState, useEffect, useRef, useCallback, useMemo } = React;
const { createRoot } = ReactDOM;

const IS_LOCAL = window.location.hostname === "localhost";
const EVENTS_API_URL = IS_LOCAL ? "http://localhost:8080" : "";

const RANGES = [
  { label: "24h", days: 1 },
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
];

// --- Helpers ---

const fetchStats = async (endpoint, days, extra = {}) => {
  const params = new URLSearchParams({ days: String(days), ...extra });
  try {
    const res = await fetch(`${EVENTS_API_URL}/stats/${endpoint}?${params}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
};

const getCSSVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const truncateUrl = (url, max = 45) => {
  try {
    const u = new URL(url);
    const short =
      u.hostname.replace("www.", "") + u.pathname.replace(/\/$/, "");
    return short.length > max ? short.slice(0, max) + "\u2026" : short;
  } catch {
    return url.length > max ? url.slice(0, max) + "\u2026" : url;
  }
};

const formatNum = (n) => {
  if (n == null) return "\u2014";
  if (n >= 10000) return (n / 1000).toFixed(1) + "k";
  return Number(n).toLocaleString();
};

// --- Chart theming ---

const getChartColors = () => ({
  accent: getCSSVar("--accent-primary") || "#10b981",
  accent2: getCSSVar("--accent-secondary") || "#06d6a0",
  text: getCSSVar("--text-muted") || "#6b6b7b",
  textSec: getCSSVar("--text-secondary") || "#a0a0b0",
  grid: getCSSVar("--border-subtle") || "rgba(255,255,255,0.06)",
});

const PALETTE = [
  "#10b981",
  "#3b82f6",
  "#f59e0b",
  "#8b5cf6",
  "#ef4444",
  "#06d6a0",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#6366f1",
  "#84cc16",
  "#0ea5e9",
];

// --- SVG Icons ---

const SearchIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const ClickIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M15 15l-2 5L9 9l11 4-5 2z" />
    <path d="M22 22l-5-10" />
  </svg>
);

const PercentIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="19" y1="5" x2="5" y2="19" />
    <circle cx="6.5" cy="6.5" r="2.5" />
    <circle cx="17.5" cy="17.5" r="2.5" />
  </svg>
);

const ClockIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

const ArrowLeftIcon = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const ChartLineIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);

const BarChartIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="12" y1="20" x2="12" y2="10" />
    <line x1="18" y1="20" x2="18" y2="4" />
    <line x1="6" y1="20" x2="6" y2="16" />
  </svg>
);

const PieChartIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
    <path d="M22 12A10 10 0 0 0 12 2v10z" />
  </svg>
);

const FolderIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>
);

const LinkIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </svg>
);

// --- Theme toggle (same as main app) ---

const ThemeToggle = ({ theme, onToggle }) => (
  <button className="theme-toggle" onClick={onToggle} aria-label="Toggle theme">
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {theme === "dark" ? (
        <React.Fragment>
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </React.Fragment>
      ) : (
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      )}
    </svg>
  </button>
);

// --- Chart wrapper ---

const ChartCanvas = ({ type, data, options, height = 260 }) => {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(canvasRef.current, {
      type,
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600, easing: "easeOutQuart" },
        ...options,
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [type, data, options, height]);

  return (
    <div className="chart-body" style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
};

// --- KPI card ---

const KpiCard = ({ icon, label, value, sub }) => (
  <div className="kpi-card">
    <div className="kpi-icon">{icon}</div>
    <div className="kpi-label">{label}</div>
    <div className="kpi-value">{value}</div>
    {sub && <div className="kpi-sub">{sub}</div>}
  </div>
);

// --- Ranked table ---

const RankedTable = ({
  icon,
  title,
  rows,
  labelKey,
  countKey,
  formatLabel,
}) => (
  <div className="dash-table-card">
    <div className="table-header">
      <span style={{ color: "var(--accent-primary)" }}>{icon}</span>
      <span className="table-title">{title}</span>
      <span className="table-count">{rows.length} items</span>
    </div>
    {rows.length === 0 ? (
      <div className="chart-empty">No data yet</div>
    ) : (
      <table className="dash-table">
        <thead>
          <tr>
            <th style={{ width: 40 }}>#</th>
            <th>
              {labelKey === "folder_name"
                ? "Folder"
                : labelKey === "query"
                  ? "Query"
                  : "Document"}
            </th>
            <th className="num">Count</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const maxCount = rows[0]?.[countKey] || 1;
            return (
              <tr key={i}>
                <td>
                  <span
                    className={`table-rank ${i < 3 ? "table-rank-top" : ""}`}
                  >
                    {i + 1}
                  </span>
                </td>
                <td>
                  {formatLabel ? formatLabel(row[labelKey]) : row[labelKey]}
                  <div
                    className="table-bar"
                    style={{ width: `${(row[countKey] / maxCount) * 100}%` }}
                  />
                </td>
                <td className="num">{row[countKey].toLocaleString()}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    )}
  </div>
);

// --- Main Dashboard ---

const Dashboard = () => {
  const [days, setDays] = useState(7);
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute("data-theme") || "dark",
  );
  const [overview, setOverview] = useState(null);
  const [activity, setActivity] = useState([]);
  const [topQueries, setTopQueries] = useState([]);
  const [topClicks, setTopClicks] = useState([]);
  const [sourcesData, setSourcesData] = useState([]);
  const [foldersData, setFoldersData] = useState([]);
  const [loading, setLoading] = useState(true);

  const toggleTheme = useCallback(() => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    setTheme(next);
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      fetchStats("overview", days),
      fetchStats("activity", days),
      fetchStats("top-queries", days, { limit: "15" }),
      fetchStats("top-clicks", days, { limit: "15" }),
      fetchStats("sources", days),
      fetchStats("folders", days, { limit: "20" }),
    ]).then(([ov, act, tq, tc, src, fld]) => {
      if (cancelled) return;
      setOverview(ov);
      setActivity(act || []);
      setTopQueries(tq || []);
      setTopClicks(tc || []);
      setSourcesData(src || []);
      setFoldersData(fld || []);
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [days]);

  // Build chart configs from current theme
  const chartConfigs = useMemo(() => {
    const c = getChartColors();
    const axis = {
      ticks: { color: c.text, font: { size: 11, family: "Inter" } },
      grid: { color: c.grid, drawBorder: false },
      border: { display: false },
    };
    return { c, axis };
  }, [theme]);

  const { c, axis } = chartConfigs;

  const activityData = useMemo(
    () => ({
      labels: activity.map((a) => {
        const d = new Date(a.period);
        return days <= 2
          ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : d.toLocaleDateString([], { month: "short", day: "numeric" });
      }),
      datasets: [
        {
          label: "Searches",
          data: activity.map((a) => a.searches),
          borderColor: c.accent,
          backgroundColor: c.accent + "18",
          fill: true,
          tension: 0.4,
          pointRadius: days <= 2 ? 0 : 3,
          pointHoverRadius: 5,
          pointBackgroundColor: c.accent,
          borderWidth: 2,
        },
        {
          label: "Clicks",
          data: activity.map((a) => a.clicks),
          borderColor: c.accent2,
          backgroundColor: c.accent2 + "18",
          fill: true,
          tension: 0.4,
          pointRadius: days <= 2 ? 0 : 3,
          pointHoverRadius: 5,
          pointBackgroundColor: c.accent2,
          borderWidth: 2,
        },
      ],
    }),
    [activity, days, c],
  );

  const activityOptions = useMemo(
    () => ({
      scales: {
        x: {
          ...axis,
          ticks: { ...axis.ticks, maxRotation: 0, maxTicksLimit: 12 },
        },
        y: { ...axis, beginAtZero: true },
      },
      plugins: {
        legend: {
          labels: {
            color: c.textSec,
            font: { size: 11, family: "Inter" },
            usePointStyle: true,
            pointStyle: "circle",
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: "rgba(0,0,0,0.8)",
          titleFont: { family: "Inter", size: 12 },
          bodyFont: { family: "Inter", size: 11 },
          cornerRadius: 8,
          padding: 10,
        },
      },
      interaction: { intersect: false, mode: "index" },
    }),
    [axis, c],
  );

  const queriesData = useMemo(
    () => ({
      labels: topQueries.map((q) =>
        q.query.length > 30 ? q.query.slice(0, 30) + "\u2026" : q.query,
      ),
      datasets: [
        {
          data: topQueries.map((q) => q.count),
          backgroundColor: c.accent + "bb",
          hoverBackgroundColor: c.accent,
          borderRadius: 4,
          borderSkipped: false,
        },
      ],
    }),
    [topQueries, c],
  );

  const barOptions = useMemo(
    () => ({
      indexAxis: "y",
      scales: {
        x: {
          ...axis,
          beginAtZero: true,
          ticks: { ...axis.ticks, maxTicksLimit: 6 },
        },
        y: {
          ...axis,
          ticks: { ...axis.ticks, font: { size: 11, family: "Inter" } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(0,0,0,0.8)",
          titleFont: { family: "Inter", size: 12 },
          bodyFont: { family: "Inter", size: 11 },
          cornerRadius: 8,
          padding: 10,
        },
      },
    }),
    [axis],
  );

  const sourcesChartData = useMemo(
    () => ({
      labels: sourcesData.map((s) => s.source_key),
      datasets: [
        {
          data: sourcesData.map((s) => s.count),
          backgroundColor: PALETTE.slice(0, sourcesData.length),
          hoverOffset: 6,
          borderWidth: 0,
        },
      ],
    }),
    [sourcesData],
  );

  const doughnutOptions = useMemo(
    () => ({
      cutout: "65%",
      plugins: {
        legend: {
          position: "right",
          labels: {
            color: c.textSec,
            font: { size: 11, family: "Inter" },
            usePointStyle: true,
            pointStyle: "circle",
            padding: 14,
          },
        },
        tooltip: {
          backgroundColor: "rgba(0,0,0,0.8)",
          titleFont: { family: "Inter", size: 12 },
          bodyFont: { family: "Inter", size: 11 },
          cornerRadius: 8,
          padding: 10,
        },
      },
    }),
    [c],
  );

  const rangeLabel = RANGES.find((r) => r.days === days)?.label || days + "d";

  return (
    <div className="dash">
      {/* Header */}
      <div className="dash-header">
        <div className="dash-title-row">
          <h1 className="dash-title">Analytics</h1>
          <a href="index.html" className="dash-back">
            <ArrowLeftIcon /> Search
          </a>
        </div>
        <div className="dash-controls">
          {RANGES.map((r) => (
            <button
              key={r.days}
              className={`range-chip ${days === r.days ? "active" : ""}`}
              onClick={() => setDays(r.days)}
            >
              {r.label}
            </button>
          ))}
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </div>

      {loading ? (
        <div className="dash-loading">
          <div className="dash-spinner" />
        </div>
      ) : (
        <React.Fragment>
          {/* KPI row */}
          <div className="dash-kpis">
            <KpiCard
              icon={<SearchIcon />}
              label="Searches"
              value={formatNum(overview?.searches)}
              sub={`in the last ${rangeLabel}`}
            />
            <KpiCard
              icon={<ClickIcon />}
              label="Clicks"
              value={formatNum(overview?.clicks)}
              sub={`${formatNum(overview?.sessions)} sessions`}
            />
            <KpiCard
              icon={<PercentIcon />}
              label="Click-through Rate"
              value={overview ? overview.ctr.toFixed(1) + "%" : "\u2014"}
              sub="clicks / searches"
            />
            <KpiCard
              icon={<ClockIcon />}
              label="Avg Latency"
              value={
                overview
                  ? Math.round(overview.avg_latency_ms) + " ms"
                  : "\u2014"
              }
              sub="search response time"
            />
          </div>

          {/* Charts */}
          <div className="dash-charts">
            <div className="chart-card chart-card-wide">
              <div className="chart-header">
                <span style={{ color: "var(--accent-primary)" }}>
                  <ChartLineIcon />
                </span>
                <span className="chart-title">Activity Over Time</span>
              </div>
              {activity.length > 0 ? (
                <ChartCanvas
                  type="line"
                  data={activityData}
                  options={activityOptions}
                  height={240}
                />
              ) : (
                <div className="chart-empty">No activity data yet</div>
              )}
            </div>

            <div className="chart-card">
              <div className="chart-header">
                <span style={{ color: "var(--accent-primary)" }}>
                  <BarChartIcon />
                </span>
                <span className="chart-title">Top Queries</span>
              </div>
              {topQueries.length > 0 ? (
                <ChartCanvas
                  type="bar"
                  data={queriesData}
                  options={barOptions}
                  height={Math.max(200, topQueries.length * 28)}
                />
              ) : (
                <div className="chart-empty">No query data yet</div>
              )}
            </div>

            <div className="chart-card">
              <div className="chart-header">
                <span style={{ color: "var(--accent-primary)" }}>
                  <PieChartIcon />
                </span>
                <span className="chart-title">Source Filter Usage</span>
              </div>
              {sourcesData.length > 0 ? (
                <ChartCanvas
                  type="doughnut"
                  data={sourcesChartData}
                  options={doughnutOptions}
                />
              ) : (
                <div className="chart-empty">No source filter data yet</div>
              )}
            </div>
          </div>

          {/* Tables */}
          <div className="dash-tables">
            <RankedTable
              icon={<LinkIcon />}
              title="Top Clicked Documents"
              rows={topClicks}
              labelKey="doc_url"
              countKey="count"
              formatLabel={(url) => truncateUrl(url)}
            />
            <RankedTable
              icon={<FolderIcon />}
              title="Popular Folders"
              rows={foldersData}
              labelKey="folder_name"
              countKey="count"
            />
          </div>

          {/* Footer */}
          <div className="dash-footer">
            {formatNum(overview?.sessions)} unique sessions &middot; Last{" "}
            {rangeLabel}
          </div>
        </React.Fragment>
      )}
    </div>
  );
};

// --- Mount ---
createRoot(document.getElementById("dashboard")).render(<Dashboard />);
