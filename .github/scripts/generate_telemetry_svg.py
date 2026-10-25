#!/usr/bin/env python3
"""
Generates a live Quantitative Forecasting & Contribution Telemetry SVG card
(`assets/quant-telemetry.svg`) from real GitHub data for @ongyiumark.

Data Pipeline:
1. Extracts 52-week daily contribution counts and repository language bytes
   via GitHub GraphQL API (when GITHUB_TOKEN / METRICS_TOKEN is set) or
   public GitHub endpoints.
2. Aggregates weekly contribution series y_t and fits an L2-Regularized
   Autoregressive & Harmonic Dynamic Factor Model (Ridge AR-DFM):
       beta_hat = (X^T X + lambda * I)^(-1) X^T y
3. Projects an 8-week out-of-sample nowcast/forecast horizon with analytical
   95% prediction intervals.
4. Outputs a self-contained, GitHub-Camo-compatible SVG with CSS animations.
"""

import datetime
import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

USERNAME = os.environ.get("GITHUB_USERNAME", "ongyiumark")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "assets/quant-telemetry.svg")

LANGUAGE_COLORS: Dict[str, str] = {
    "Python": "#3572A5",
    "TypeScript": "#3178c6",
    "JavaScript": "#f1e05a",
    "C++": "#f34b7d",
    "C": "#555555",
    "SQL": "#e38c00",
    "PLpgSQL": "#336790",
    "HTML": "#e34c26",
    "CSS": "#563d7c",
    "Jupyter Notebook": "#DA5B0B",
    "R": "#198CE7",
    "Shell": "#89e051",
    "Rust": "#dea584",
    "Go": "#00ADD8",
}


def http_get(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    req_headers = {
        "User-Agent": "ongyiumark-telemetry-generator/1.0",
        "Accept": "application/vnd.github+json",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_via_graphql(username: str, token: str, use_viewer: bool = False) -> Tuple[List[Tuple[str, int]], Dict[str, int], int]:
    """Fetch contribution calendar and language byte breakdown via GitHub GraphQL API."""
    root_field = "viewer" if use_viewer else "user(login: $login)"
    query_header = "query" if use_viewer else "query($login: String!)"
    query = f"""
    {query_header} {{
      target: {root_field} {{
        contributionsCollection {{
          contributionCalendar {{
            totalContributions
            weeks {{
              contributionDays {{
                date
                contributionCount
              }}
            }}
          }}
        }}
        repositories(first: 100, ownerAffiliations: OWNER, isFork: false, orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
          nodes {{
            name
            languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
              edges {{
                size
                node {{
                  name
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    variables = {} if use_viewer else {"login": username}
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ongyiumark-telemetry-generator/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "errors" in data and not data.get("data", {}).get("target"):
        raise RuntimeError(f"GraphQL error: {data['errors']}")

    user_data = data["data"]["target"]
    cal = user_data["contributionsCollection"]["contributionCalendar"]
    total_contributions = int(cal["totalContributions"])

    daily: List[Tuple[str, int]] = []
    for week in cal["weeks"]:
        for day in week["contributionDays"]:
            daily.append((day["date"], int(day["contributionCount"])))

    lang_bytes: Dict[str, int] = {}
    for repo in user_data["repositories"]["nodes"]:
        if repo["name"].lower() == username.lower():
            continue
        for edge in repo.get("languages", {}).get("edges", []):
            lname = edge["node"]["name"]
            lang_bytes[lname] = lang_bytes.get(lname, 0) + int(edge["size"])

    return daily, lang_bytes, total_contributions


def fetch_via_public(username: str) -> Tuple[List[Tuple[str, int]], Dict[str, int], int]:
    """Fallback fetcher using public GitHub profile contributions HTML and REST API."""
    daily: List[Tuple[str, int]] = []
    total_contributions = 0

    # 1. Fetch live from github.com/users/{username}/contributions first (reflects profile settings immediately)
    try:
        html_text = http_get(
            f"https://github.com/users/{username}/contributions",
            headers={"Accept": "text/html"},
        )
        date_matches = re.findall(r'data-date="(\d{4}-\d{2}-\d{2})"[^>]*id="([^"]+)"', html_text)
        tooltip_map: Dict[str, int] = {}
        for tid, text in re.findall(r'for="([^"]+)"[^>]*>([^<]+)</tool-tip>', html_text):
            m = re.search(r"(\d+)\s+contribution", text)
            tooltip_map[tid] = int(m.group(1)) if m else 0
        for d_str, tid in date_matches:
            c = tooltip_map.get(tid, 0)
            daily.append((d_str, c))
        daily.sort(key=lambda x: x[0])
        total_contributions = sum(c for _, c in daily)
    except Exception:
        pass

    if not daily:
        raw_json = http_get(f"https://github-contributions-api.jogruber.de/v4/{username}?y=last")
        parsed = json.loads(raw_json)
        for item in parsed.get("contributions", []):
            c = int(item.get("count", 0))
            daily.append((item["date"], c))
        total_contributions = sum(c for _, c in daily)

    # 2. Fetch public repositories for language breakdown (normalized per repo)
    lang_bytes: Dict[str, int] = {}
    try:
        repos_json = json.loads(
            http_get(f"https://api.github.com/users/{username}/repos?per_page=100&type=owner")
        )
        for repo in repos_json:
            if repo.get("fork") or repo.get("name", "").lower() == username.lower():
                continue
            lang = repo.get("language")
            if lang:
                # Weight by log2(size_kb + 2) so one repo with large data assets doesn't dominate
                size_kb = max(int(repo.get("size", 1)), 1)
                weight = int(round(math.log2(size_kb + 2) * 1000))
                lang_bytes[lang] = lang_bytes.get(lang, 0) + weight
    except Exception:
        pass

    return daily, lang_bytes, total_contributions


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        denom = M[col][col]
        if abs(denom) < 1e-12:
            denom = 1e-12
        for j in range(col, n + 1):
            M[col][j] /= denom
        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def fit_regularized_dfm_ar(
    series: List[float], horizon: int = 8, ridge_lambda: float = 1.5
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Fits an L2-Regularized Autoregressive + Dynamic Factor Model on weekly series y_t.
    Features at time t:
      - Intercept (1.0)
      - Lag 1: y_{t-1}
      - Lag 2: y_{t-2}
      - Local Momentum Factor: mean(y_{t-4 : t})
      - Annual/Quarterly Harmonic Factors: sin(2*pi*t/13), cos(2*pi*t/13)
    Returns (in_sample_fitted, forecast_mean, forecast_std, rmse).
    """
    n = len(series)
    if n < 8:
        mean_val = sum(series) / max(n, 1)
        return series[:], [mean_val] * horizon, [max(mean_val * 0.3, 1.0)] * horizon, 0.0

    # Smooth raw weekly counts with a causal 3-week exponentially weighted factor
    # so the latent activity signal captures multi-week engineering sprints cleanly.
    alpha = 0.55
    latent: List[float] = [series[0]]
    for val in series[1:]:
        latent.append(alpha * val + (1.0 - alpha) * latent[-1])

    lags = 4
    X: List[List[float]] = []
    y_target: List[float] = []

    for t in range(lags, n):
        f_intercept = 1.0
        f_lag1 = latent[t - 1]
        f_lag2 = latent[t - 2]
        f_mom = sum(latent[t - lags : t]) / float(lags)
        f_sin = math.sin(2.0 * math.pi * t / 13.0)
        f_cos = math.cos(2.0 * math.pi * t / 13.0)
        X.append([f_intercept, f_lag1, f_lag2, f_mom, f_sin, f_cos])
        y_target.append(latent[t])

    p = len(X[0])
    # Compute X^T X + lambda * I (do not regularize intercept at index 0)
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for row, y_val in zip(X, y_target):
        for i in range(p):
            Xty[i] += row[i] * y_val
            for j in range(p):
                XtX[i][j] += row[i] * row[j]
    for i in range(1, p):
        XtX[i][i] += ridge_lambda

    beta = solve_linear_system(XtX, Xty)

    # In-sample predictions and residual variance
    fitted = latent[:lags]
    sq_errors: List[float] = []
    for row, y_val in zip(X, y_target):
        pred = max(0.0, sum(b * x for b, x in zip(beta, row)))
        fitted.append(pred)
        sq_errors.append((y_val - pred) ** 2)

    dof = max(1, len(sq_errors) - p)
    sigma2 = sum(sq_errors) / dof
    rmse = math.sqrt(sum(sq_errors) / max(1, len(sq_errors)))
    base_std = max(math.sqrt(sigma2), max(latent) * 0.12, 0.8)

    # Multi-step out-of-sample projection with expanding state uncertainty
    history = latent[:]
    forecast_mean: List[float] = []
    forecast_std: List[float] = []
    phi1 = beta[1] + beta[3] / float(lags)
    psi_sum_sq = 1.0

    for h in range(1, horizon + 1):
        t_idx = n - 1 + h
        row = [
            1.0,
            history[-1],
            history[-2],
            sum(history[-lags:]) / float(lags),
            math.sin(2.0 * math.pi * t_idx / 13.0),
            math.cos(2.0 * math.pi * t_idx / 13.0),
        ]
        pred = max(0.2, sum(b * x for b, x in zip(beta, row)))
        forecast_mean.append(pred)
        history.append(pred)

        if h > 1:
            psi_sum_sq += (min(max(phi1, 0.25), 0.85)) ** (2 * (h - 1))
        forecast_std.append(base_std * math.sqrt(psi_sum_sq))

    return latent, forecast_mean, forecast_std, rmse


def aggregate_weekly(daily: List[Tuple[str, int]], num_weeks: int = 26) -> Tuple[List[str], List[float]]:
    """Aggregate daily (date, count) into the most recent `num_weeks` 7-day buckets."""
    if not daily:
        today = datetime.date.today()
        dates = [(today - datetime.timedelta(weeks=num_weeks - 1 - i)).isoformat() for i in range(num_weeks)]
        return dates, [0.0] * num_weeks

    daily_sorted = sorted(daily, key=lambda x: x[0])
    # Group from the end in 7-day chunks
    total_days = len(daily_sorted)
    weeks_dates: List[str] = []
    weeks_counts: List[float] = []

    idx = total_days
    while idx > 0 and len(weeks_counts) < num_weeks:
        start = max(0, idx - 7)
        chunk = daily_sorted[start:idx]
        weeks_dates.append(chunk[-1][0])
        weeks_counts.append(float(sum(c for _, c in chunk)))
        idx = start

    weeks_dates.reverse()
    weeks_counts.reverse()
    return weeks_dates, weeks_counts


def smooth_svg_path(
    points: List[Tuple[float, float]], min_y: float = 56.0, max_y: float = 176.0
) -> str:
    """Create a smooth, monotone-clamped cubic Bezier SVG path through the given points."""
    if not points:
        return ""
    if len(points) == 1:
        return f"M {points[0][0]:.1f} {points[0][1]:.1f}"

    path = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]

        tension = 0.18
        cp1x = p1[0] + (p2[0] - p0[0]) * tension
        cp1y = min(max_y, max(min_y, p1[1] + (p2[1] - p0[1]) * tension))
        cp2x = p2[0] - (p3[0] - p1[0]) * tension
        cp2y = min(max_y, max(min_y, p2[1] - (p3[1] - p1[1]) * tension))

        path.append(f"C {cp1x:.1f} {cp1y:.1f}, {cp2x:.1f} {cp2y:.1f}, {p2[0]:.1f} {p2[1]:.1f}")
    return " ".join(path)


def render_svg(
    week_dates: List[str],
    raw_weekly: List[float],
    latent_signal: List[float],
    forecast_mean: List[float],
    forecast_std: List[float],
    rmse: float,
    lang_bytes: Dict[str, int],
    total_year_contribs: int,
) -> str:
    width = 800
    height = 250
    pad_left = 48
    pad_right = 32
    plot_top = 56
    plot_bottom = 176
    plot_w = width - pad_left - pad_right
    plot_h = plot_bottom - plot_top

    n_obs = len(latent_signal)
    n_fc = len(forecast_mean)
    total_steps = n_obs + n_fc - 1

    upper_ci = [m + 1.96 * s for m, s in zip(forecast_mean, forecast_std)]
    lower_ci = [max(0.0, m - 1.96 * s) for m, s in zip(forecast_mean, forecast_std)]

    max_val = max(
        max(raw_weekly, default=1.0),
        max(latent_signal, default=1.0),
        max(upper_ci, default=1.0),
        4.0,
    ) * 1.15

    def to_xy(step_idx: int, val: float) -> Tuple[float, float]:
        x = pad_left + (step_idx / max(1, total_steps)) * plot_w
        clamped = min(max(val, 0.0), max_val)
        y = plot_bottom - (clamped / max_val) * plot_h
        return x, y

    obs_pts = [to_xy(i, v) for i, v in enumerate(latent_signal)]
    raw_pts = [to_xy(i, v) for i, v in enumerate(raw_weekly)]
    fc_pts = [obs_pts[-1]] + [to_xy(n_obs + j, v) for j, v in enumerate(forecast_mean)]

    ci_top_pts = [obs_pts[-1]] + [to_xy(n_obs + j, v) for j, v in enumerate(upper_ci)]
    ci_bot_pts = [obs_pts[-1]] + [to_xy(n_obs + j, v) for j, v in enumerate(lower_ci)]

    # Build closed polygon for 95% confidence interval fan
    ci_poly_coords = [f"{x:.1f},{y:.1f}" for x, y in ci_top_pts + list(reversed(ci_bot_pts))]
    ci_polygon_points = " ".join(ci_poly_coords)

    # Build area under observed curve
    obs_path_d = smooth_svg_path(obs_pts)
    obs_area_d = (
        f"{obs_path_d} L {obs_pts[-1][0]:.1f} {plot_bottom:.1f} "
        f"L {obs_pts[0][0]:.1f} {plot_bottom:.1f} Z"
    )
    fc_path_d = smooth_svg_path(fc_pts)

    nowcast_x, nowcast_y = obs_pts[-1]

    # Y-axis grid & labels
    y_ticks = [0.0, max_val * 0.5, max_val]
    grid_lines_svg = []
    for yt in y_ticks:
        _, yy = to_xy(0, yt)
        grid_lines_svg.append(
            f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width - pad_right}" y2="{yy:.1f}" '
            f'stroke="#21262d" stroke-width="1" />'
            f'<text x="{pad_left - 8}" y="{yy + 4:.1f}" fill="#6e7681" font-size="10" text-anchor="end">'
            f"{int(round(yt))}</text>"
        )

    # X-axis month markers from real dates
    x_labels_svg = []
    seen_months = set()
    for i, d_str in enumerate(week_dates):
        try:
            dt = datetime.date.fromisoformat(d_str)
            m_key = (dt.year, dt.month)
            if m_key not in seen_months and i % 4 == 0:
                seen_months.add(m_key)
                xx, _ = to_xy(i, 0)
                x_labels_svg.append(
                    f'<text x="{xx:.1f}" y="{plot_bottom + 16}" fill="#6e7681" font-size="10" text-anchor="middle">'
                    f'{dt.strftime("%b")}</text>'
                )
        except ValueError:
            pass

    fc_end_x, _ = to_xy(total_steps, 0)
    x_labels_svg.append(
        f'<text x="{fc_end_x:.1f}" y="{plot_bottom + 16}" fill="#58a6ff" font-size="10" text-anchor="end">'
        f"+{n_fc}w Horizon</text>"
    )

    # Raw weekly observation bars (subtle telemetry impulses)
    bars_svg = []
    for x, y in raw_pts:
        bar_h = max(1.5, plot_bottom - y)
        bars_svg.append(
            f'<line x1="{x:.1f}" y1="{plot_bottom:.1f}" x2="{x:.1f}" y2="{plot_bottom - bar_h:.1f}" '
            f'stroke="#238636" stroke-width="3" stroke-opacity="0.35" stroke-linecap="round" />'
        )

    # Compute top languages
    if not lang_bytes:
        lang_bytes = {"Python": 60, "TypeScript": 25, "C++": 15}
    total_bytes = max(1, sum(lang_bytes.values()))
    top_langs = sorted(lang_bytes.items(), key=lambda x: x[1], reverse=True)[:4]
    lang_spans = []
    for lname, lsize in top_langs:
        pct = (lsize / total_bytes) * 100.0
        color = LANGUAGE_COLORS.get(lname, "#58a6ff")
        lang_spans.append(
            f'<tspan fill="{color}">●</tspan> <tspan fill="#c9d1d9">{html.escape(lname)} {pct:.0f}%</tspan>'
        )
    lang_summary_svg = '  <tspan fill="#484f58">•</tspan>  '.join(lang_spans)

    updated_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    active_weeks = sum(1 for c in raw_weekly if c > 0)

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace">
  <defs>
    <linearGradient id="obsGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3fb950" stop-opacity="0.25" />
      <stop offset="100%" stop-color="#3fb950" stop-opacity="0.0" />
    </linearGradient>
    <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#58a6ff" stop-opacity="0.30" />
      <stop offset="100%" stop-color="#58a6ff" stop-opacity="0.04" />
    </linearGradient>
    <style>
      .pulse-ring {{
        animation: pulse 2.4s infinite ease-out;
        transform-origin: {nowcast_x:.1f}px {nowcast_y:.1f}px;
      }}
      @keyframes pulse {{
        0% {{ r: 4px; opacity: 0.9; }}
        100% {{ r: 14px; opacity: 0; }}
      }}
    </style>
  </defs>

  <!-- Card Background -->
  <rect width="{width}" height="{height}" rx="10" fill="#0d1117" stroke="#30363d" stroke-width="1" />

  <!-- Header -->
  <text x="24" y="30" fill="#f0f6fc" font-size="13" font-weight="bold">QUANTITATIVE ACTIVITY TELEMETRY &amp; L2-REGULARIZED NOWCAST</text>
  
  <!-- Legend -->
  <line x1="455" y1="26" x2="467" y2="26" stroke="#238636" stroke-width="3" stroke-opacity="0.7" stroke-linecap="round" />
  <text x="473" y="30" fill="#8b949e" font-size="10">Weekly Commits</text>

  <circle cx="574" cy="26" r="3.5" fill="#3fb950" />
  <text x="582" y="30" fill="#8b949e" font-size="10">Filtered Signal</text>

  <circle cx="684" cy="26" r="3.5" fill="#58a6ff" />
  <text x="692" y="30" fill="#8b949e" font-size="10">95% CI Forecast</text>

  <!-- Grid Lines -->
  {"".join(grid_lines_svg)}

  <!-- Raw Weekly Impulse Bars -->
  {"".join(bars_svg)}

  <!-- Observed Latent Signal Area & Curve -->
  <path d="{obs_area_d}" fill="url(#obsGrad)" />
  <path d="{obs_path_d}" fill="none" stroke="#3fb950" stroke-width="2.2" />

  <!-- 95% Confidence Interval Fan & Forecast Trajectory -->
  <polygon points="{ci_polygon_points}" fill="url(#ciGrad)" />
  <path d="{fc_path_d}" fill="none" stroke="#58a6ff" stroke-width="2.2" stroke-dasharray="5,4" />

  <!-- Nowcast Vertical Horizon Line & Animated Marker -->
  <line x1="{nowcast_x:.1f}" y1="{plot_top - 6}" x2="{nowcast_x:.1f}" y2="{plot_bottom}" stroke="#8b949e" stroke-width="1" stroke-dasharray="3,3" />
  <circle cx="{nowcast_x:.1f}" cy="{nowcast_y:.1f}" r="4" fill="none" stroke="#3fb950" stroke-width="2" class="pulse-ring" />
  <circle cx="{nowcast_x:.1f}" cy="{nowcast_y:.1f}" r="4" fill="#0d1117" stroke="#3fb950" stroke-width="2.2" />
  <text x="{nowcast_x + 6:.1f}" y="{plot_top + 8}" fill="#8b949e" font-size="10">t₀ (Nowcast)</text>

  <!-- X-Axis Labels -->
  {"".join(x_labels_svg)}

  <!-- Bottom Telemetry Bar -->
  <rect x="20" y="202" width="{width - 40}" height="32" rx="6" fill="#161b22" stroke="#21262d" stroke-width="1" />
  <text x="32" y="222" font-size="11">{lang_summary_svg}</text>
  <text x="{width - 32}" y="222" fill="#8b949e" font-size="10" text-anchor="end">52w: <tspan fill="#3fb950" font-weight="bold">{total_year_contribs}</tspan> contribs ({active_weeks}/{len(raw_weekly)}w active) • RMSE={rmse:.2f} • {updated_utc}</text>
</svg>
"""


def main() -> None:
    metrics_token = os.environ.get("METRICS_TOKEN")
    token = metrics_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    daily: List[Tuple[str, int]] = []
    lang_bytes: Dict[str, int] = {}
    total_contribs = 0

    if token:
        try:
            daily, lang_bytes, total_contribs = fetch_via_graphql(
                USERNAME, token, use_viewer=bool(metrics_token)
            )
            print(f"Fetched live GraphQL telemetry for @{USERNAME} ({len(daily)} days, {total_contribs} total contributions).")
        except Exception as exc:
            print(f"GraphQL fetch failed ({exc}), falling back to public endpoints...")

    if not daily:
        daily, lang_bytes, total_contribs = fetch_via_public(USERNAME)
        print(f"Fetched live public telemetry for @{USERNAME} ({len(daily)} days, {total_contribs} total contributions).")

    week_dates, raw_weekly = aggregate_weekly(daily, num_weeks=52)
    latent_signal, forecast_mean, forecast_std, rmse = fit_regularized_dfm_ar(
        raw_weekly, horizon=8, ridge_lambda=1.5
    )

    svg_content = render_svg(
        week_dates=week_dates,
        raw_weekly=raw_weekly,
        latent_signal=latent_signal,
        forecast_mean=forecast_mean,
        forecast_std=forecast_std,
        rmse=rmse,
        lang_bytes=lang_bytes,
        total_year_contribs=total_contribs,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Wrote live SVG card to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
