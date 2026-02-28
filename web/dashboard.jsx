const { useState, useEffect, useRef, useCallback } = React;
const { createRoot } = ReactDOM;

const EVENTS_API_URL = "http://localhost:3002";

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

const truncateUrl = (url, max = 50) => {
  try {
    const u = new URL(url);
    const short = u.hostname + u.pathname;
    return short.length > max ? short.slice(0, max) + "..." : short;
  } catch {
    return url.length > max ? url.slice(0, max) + "..." : url;
  }
};

// --- Chart theme colors ---

const getChartColors = () => ({
  accent: getCSSVar("--accent-primary") || "#10b981",
  accent2: getCSSVar("--accent-secondary") || "#06d6a0",
  text: getCSSVar("--text-secondary") || "#a0a0b0",
  grid: getCSSVar("--border-subtle") || "rgba(255,255,255,0.06)",
  bg: getCSSVar("--bg-secondary") || "#12121a",
});

const CHART_PALETTE = [
  "#10b981",
  "#06d6a0",
  "#3b82f6",
  "#8b5cf6",
  "#f59e0b",
  "#ef4444",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#6366f1",
  "#84cc16",
  "#0ea5e9",
  "#e879f9",
  "#fb923c",
  "#22d3ee",
];

// --- Inline styles (dashboard-specific, not in style.css) ---

const styles = {
  container: {
    maxWidth: 1200,
    margin: "0 auto",
    padding: "24px 20px 60px",
    fontFamily: "Inter, system-ui, sans-serif",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 32,
  },
  titleRow: {
    display: "flex",
    alignItems: "center",
    gap: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    color: "var(--text-primary)",
    margin: 0,
  },
  backLink: {
    fontSize: 13,
    color: "var(--accent-primary)",
    textDecoration: "none",
    padding: "4px 10px",
    borderRadius: "var(--radius-sm)",
    border: "1px solid var(--border-subtle)",
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  rangeBtn: (active) => ({
    padding: "6px 14px",
    borderRadius: "var(--radius-sm)",
    border:
      "1px solid " +
      (active ? "var(--accent-primary)" : "var(--border-subtle)"),
    background: active ? "var(--accent-bg-subtle)" : "transparent",
    color: active ? "var(--accent-primary)" : "var(--text-secondary)",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
    transition: "var(--transition-fast)",
  }),
  themeBtn: {
    padding: "6px 10px",
    borderRadius: "var(--radius-sm)",
    border: "1px solid var(--border-subtle)",
    background: "transparent",
    color: "var(--text-secondary)",
    cursor: "pointer",
    fontSize: 16,
    marginLeft: 4,
  },
  kpiGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: 16,
    marginBottom: 32,
  },
  kpiCard: {
    background: "var(--bg-glass)",
    border: "1px solid var(--border-subtle)",
    borderRadius: "var(--radius-md)",
    padding: "20px 24px",
    backdropFilter: "blur(12px)",
    boxShadow: "var(--shadow-sm)",
  },
  kpiLabel: {
    fontSize: 12,
    fontWeight: 500,
    color: "var(--text-secondary)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: 6,
  },
  kpiValue: {
    fontSize: 28,
    fontWeight: 700,
    color: "var(--text-primary)",
  },
  chartGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(480px, 1fr))",
    gap: 20,
    marginBottom: 32,
  },
  chartCard: {
    background: "var(--bg-glass)",
    border: "1px solid var(--border-subtle)",
    borderRadius: "var(--radius-md)",
    padding: 20,
    backdropFilter: "blur(12px)",
    boxShadow: "var(--shadow-sm)",
  },
  chartTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: "var(--text-primary)",
    marginBottom: 16,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },
  th: {
    textAlign: "left",
    padding: "8px 12px",
    color: "var(--text-secondary)",
    borderBottom: "1px solid var(--border-subtle)",
    fontWeight: 600,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  td: {
    padding: "8px 12px",
    color: "var(--text-primary)",
    borderBottom: "1px solid var(--border-subtle)",
  },
  empty: {
    textAlign: "center",
    padding: 40,
    color: "var(--text-muted)",
    fontSize: 14,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 600,
    color: "var(--text-primary)",
    marginBottom: 16,
  },
  tableSection: {
    background: "var(--bg-glass)",
    border: "1px solid var(--border-subtle)",
    borderRadius: "var(--radius-md)",
    padding: 20,
    backdropFilter: "blur(12px)",
    boxShadow: "var(--shadow-sm)",
    marginBottom: 20,
  },
};

// --- Chart component (canvas wrapper) ---

const ChartCanvas = ({ type, data, options, title }) => {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new Chart(canvasRef.current, {
      type,
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        ...options,
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [type, data, options]);

  return (
    <div style={styles.chartCard}>
      <div style={styles.chartTitle}>{title}</div>
      <div style={{ height: 280 }}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
};

// --- KPI Card ---

const KpiCard = ({ label, value }) => (
  <div style={styles.kpiCard}>
    <div style={styles.kpiLabel}>{label}</div>
    <div style={styles.kpiValue}>{value}</div>
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

  // Fetch all data when days or theme changes
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
  }, [days, theme]);

  // Chart options
  const colors = getChartColors();

  const axisDefaults = {
    ticks: { color: colors.text, font: { size: 11 } },
    grid: { color: colors.grid },
    border: { color: colors.grid },
  };

  const activityChartData = {
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
        borderColor: colors.accent,
        backgroundColor: colors.accent + "20",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
      {
        label: "Clicks",
        data: activity.map((a) => a.clicks),
        borderColor: colors.accent2,
        backgroundColor: colors.accent2 + "20",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
    ],
  };

  const activityOptions = {
    scales: { x: axisDefaults, y: { ...axisDefaults, beginAtZero: true } },
    plugins: {
      legend: { labels: { color: colors.text, font: { size: 12 } } },
    },
  };

  const queriesChartData = {
    labels: topQueries.map((q) =>
      q.query.length > 35 ? q.query.slice(0, 35) + "..." : q.query,
    ),
    datasets: [
      {
        label: "Searches",
        data: topQueries.map((q) => q.count),
        backgroundColor: colors.accent + "cc",
        borderRadius: 4,
      },
    ],
  };

  const horizontalBarOptions = {
    indexAxis: "y",
    scales: {
      x: { ...axisDefaults, beginAtZero: true },
      y: {
        ...axisDefaults,
        ticks: { ...axisDefaults.ticks, font: { size: 11 } },
      },
    },
    plugins: { legend: { display: false } },
  };

  const clicksChartData = {
    labels: topClicks.map((c) => truncateUrl(c.doc_url)),
    datasets: [
      {
        label: "Clicks",
        data: topClicks.map((c) => c.count),
        backgroundColor: colors.accent2 + "cc",
        borderRadius: 4,
      },
    ],
  };

  const sourcesChartData = {
    labels: sourcesData.map((s) => s.source_key),
    datasets: [
      {
        data: sourcesData.map((s) => s.count),
        backgroundColor: CHART_PALETTE.slice(0, sourcesData.length),
        borderWidth: 0,
      },
    ],
  };

  const doughnutOptions = {
    plugins: {
      legend: {
        position: "right",
        labels: { color: colors.text, font: { size: 12 }, padding: 12 },
      },
    },
  };

  const formatNum = (n) => {
    if (n == null) return "--";
    return Number(n).toLocaleString();
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.titleRow}>
          <h1 style={styles.title}>Analytics</h1>
          <a href="index.html" style={styles.backLink}>
            Back to Search
          </a>
        </div>
        <div style={styles.controls}>
          {RANGES.map((r) => (
            <button
              key={r.days}
              style={styles.rangeBtn(days === r.days)}
              onClick={() => setDays(r.days)}
            >
              {r.label}
            </button>
          ))}
          <button style={styles.themeBtn} onClick={toggleTheme}>
            {theme === "dark" ? "\u2600\uFE0F" : "\uD83C\uDF19"}
          </button>
        </div>
      </div>

      {/* KPI row */}
      <div style={styles.kpiGrid}>
        <KpiCard label="Total Searches" value={formatNum(overview?.searches)} />
        <KpiCard label="Total Clicks" value={formatNum(overview?.clicks)} />
        <KpiCard
          label="Click-through Rate"
          value={overview ? overview.ctr.toFixed(1) + "%" : "--"}
        />
        <KpiCard
          label="Avg Latency"
          value={overview ? overview.avg_latency_ms.toFixed(0) + " ms" : "--"}
        />
      </div>

      {/* Charts grid */}
      <div style={styles.chartGrid}>
        {activity.length > 0 ? (
          <ChartCanvas
            type="line"
            data={activityChartData}
            options={activityOptions}
            title="Activity Over Time"
          />
        ) : (
          <div style={{ ...styles.chartCard, ...styles.empty }}>
            No activity data yet
          </div>
        )}

        {topQueries.length > 0 ? (
          <ChartCanvas
            type="bar"
            data={queriesChartData}
            options={horizontalBarOptions}
            title="Top Queries"
          />
        ) : (
          <div style={{ ...styles.chartCard, ...styles.empty }}>
            No query data yet
          </div>
        )}

        {topClicks.length > 0 ? (
          <ChartCanvas
            type="bar"
            data={clicksChartData}
            options={horizontalBarOptions}
            title="Top Clicked Documents"
          />
        ) : (
          <div style={{ ...styles.chartCard, ...styles.empty }}>
            No click data yet
          </div>
        )}

        {sourcesData.length > 0 ? (
          <ChartCanvas
            type="doughnut"
            data={sourcesChartData}
            options={doughnutOptions}
            title="Source Filter Usage"
          />
        ) : (
          <div style={{ ...styles.chartCard, ...styles.empty }}>
            No source filter data yet
          </div>
        )}
      </div>

      {/* Folders table */}
      {foldersData.length > 0 && (
        <div style={styles.tableSection}>
          <div style={styles.sectionTitle}>Popular Folders</div>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Folder</th>
                <th style={{ ...styles.th, textAlign: "right" }}>Views</th>
              </tr>
            </thead>
            <tbody>
              {foldersData.map((f, i) => (
                <tr key={i}>
                  <td style={styles.td}>{f.folder_name}</td>
                  <td style={{ ...styles.td, textAlign: "right" }}>
                    {f.count.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Sessions count */}
      {overview && (
        <div
          style={{
            textAlign: "center",
            color: "var(--text-muted)",
            fontSize: 12,
            marginTop: 24,
          }}
        >
          {formatNum(overview.sessions)} unique sessions in the last{" "}
          {RANGES.find((r) => r.days === days)?.label || days + "d"}
        </div>
      )}
    </div>
  );
};

// --- Mount ---

const root = createRoot(document.getElementById("dashboard"));
root.render(<Dashboard />);
