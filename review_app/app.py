"""Review tool for the expert validation. Start with: streamlit run review_app/app.py

Experts log in with a personal access code. They see one segment at a time, without the
driver and without the reason why the segment was selected, and answer three questions.
The folder with the review package is taken from the configuration, or from the environment
variable KM_REVIEW_DIR when the tool runs on a server.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from km import review  # noqa: E402
from km.advice import ADVICE_NAMES, TYPE_NAMES  # noqa: E402
from km.config import Paths, load_config  # noqa: E402
from km.labelling import SCHEMA, TYPES  # noqa: E402
from review_app.texts import T, t  # noqa: E402

LAP_COLOUR, REF_COLOUR = "#1f4e79", "#b8860b"


def review_dir() -> Path:
    if os.environ.get("KM_REVIEW_DIR"):
        return Path(os.environ["KM_REVIEW_DIR"])
    return Paths.from_config(load_config(os.environ.get("KM_CONFIG"))).review


def load_budget() -> int:
    """Number of corners an expert may select per session (B). On a server it is set with KM_BUDGET."""
    try:
        return int(load_config(os.environ.get("KM_CONFIG"))["review"]["budget"])
    except Exception:
        return 3


@st.cache_data(show_spinner=False)
def load_package(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = Path(folder)
    return (pd.read_parquet(folder / "items.parquet"), pd.read_parquet(folder / "assignments.parquet"),
            pd.read_parquet(folder / "traces.parquet"))


def trace_figure(trace: pd.DataFrame, lang: str) -> go.Figure:
    rows = [("speed", 1.0), ("gas", 100.0), ("brake", 100.0), ("steer", 1.0)]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.3, 0.17, 0.17, 0.17, 0.19])
    x = trace["dist_m"]
    for k, (name, scale) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=x, y=trace[f"ref_{name}"] * scale, name=t("reference", lang), legendgroup="ref",
                                 showlegend=k == 1, line=dict(color=REF_COLOUR, width=2, dash="dash")), row=k, col=1)
        fig.add_trace(go.Scatter(x=x, y=trace[name] * scale, name=t("this_lap", lang), legendgroup="lap",
                                 showlegend=k == 1, line=dict(color=LAP_COLOUR, width=2)), row=k, col=1)
        fig.update_yaxes(title_text=t(name, lang), row=k, col=1, title_font_size=11)
    fig.add_trace(go.Scatter(x=x, y=trace["delta_t"], showlegend=False, line=dict(color="#4d4d4d", width=2)), row=5, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#999999", row=5, col=1)
    fig.update_yaxes(title_text=t("delta", lang), row=5, col=1, title_font_size=11)
    fig.update_xaxes(title_text=t("distance", lang), row=5, col=1)
    inside = trace.loc[trace["in_segment"], "dist_m"]
    if len(inside):
        fig.add_vrect(x0=float(inside.min()), x1=float(inside.max()), fillcolor="#1f4e79", opacity=0.07, line_width=0)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def page_login(folder: Path, lang: str) -> None:
    st.title(t("app_title", lang))
    st.write(t("login_intro", lang))
    with st.form("login"):
        code = st.text_input(t("code", lang), type="password")
        if st.form_submit_button(t("login", lang)):
            user = review.check_code(folder, code)
            if user is None:
                st.error(t("bad_code", lang))
            else:
                st.session_state["user"] = user
                st.rerun()


def page_intro(folder: Path, user: dict, lang: str) -> None:
    st.title(t("app_title", lang))
    st.markdown(t("intro", lang))
    st.subheader(t("schema_title", lang))
    st.table(pd.DataFrame({
        "": TYPES,
        t("col_type", lang): [TYPE_NAMES[lang][k] for k in TYPES],
        t("col_phase", lang): [T["phases"][lang][SCHEMA[k]["phase"]] for k in TYPES],
        t("col_advice", lang): [ADVICE_NAMES[lang][k] for k in TYPES],
    }).set_index(""))
    if user["role"] != "expert":
        return
    st.subheader(t("profile_title", lang))
    st.caption(t("profile_intro", lang))
    with st.form("profile"):
        roles = st.multiselect(t("p_role", lang), T["p_roles"]["en"],
                               format_func=lambda v: T["p_roles"][lang][T["p_roles"]["en"].index(v)])
        years = st.number_input(t("p_years", lang), 0, 60, 0)
        coach_years = st.number_input(t("p_coach_years", lang), 0, 60, 0)
        telemetry = st.selectbox(t("p_telemetry", lang), T["p_telemetry_opts"]["en"],
                                 format_func=lambda v: T["p_telemetry_opts"][lang][T["p_telemetry_opts"]["en"].index(v)])
        if st.form_submit_button(t("save", lang)):
            review.save_profile(folder, user["expert_id"], json.dumps(
                dict(roles=roles, years=int(years), coach_years=int(coach_years), telemetry=telemetry)))
            st.success(t("saved", lang))


def page_review(folder: Path, user: dict, lang: str) -> None:
    items, assignments, traces = load_package(str(folder))
    mine = assignments[assignments["expert_id"] == user["expert_id"]].sort_values("position")
    if mine.empty:
        st.info(t("dash_empty", lang))
        return
    order = mine["window_id"].tolist()
    responses = review.load_responses(folder)
    done = responses[responses["expert_id"] == user["expert_id"]].set_index("window_id")
    open_positions = [k for k, w in enumerate(order) if w not in done.index]
    if "position" not in st.session_state:
        st.session_state["position"] = open_positions[0] if open_positions else len(order) - 1
    pos = min(st.session_state["position"], len(order) - 1)
    window_id = order[pos]
    if st.session_state.get("timer_for") != window_id:
        st.session_state["timer_for"], st.session_state["timer"] = window_id, time.time()

    st.progress(len(done) / len(order), text=f"{t('progress', lang, i=pos + 1, n=len(order))}   ({len(done)}/{len(order)})")
    if not open_positions:
        st.success(t("all_done", lang))
    item = items[items["window_id"] == window_id].iloc[0]
    kind = item["shown_type"]

    st.subheader(f"{item['circuit'].replace('_', ' ').title()}, {item['segment_name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric(t("seg_time", lang), f"{item['seg_time']:.2f} s")
    c2.metric(t("ref_time", lang), f"{item['ref_seg_time']:.2f} s")
    c3.metric(t("difference", lang), f"{item['time_loss']:+.2f} s")
    st.plotly_chart(trace_figure(traces[traces["window_id"] == window_id], lang), width="stretch")
    st.caption(t("segment_area", lang))

    st.markdown(f"**{t('assigned_type', lang)}:** {kind} {TYPE_NAMES[lang][kind]}")
    st.markdown(f"**{t('assigned_advice', lang)}:** {ADVICE_NAMES[lang][kind]}")
    st.info(item[f"advice_{lang}"])

    previous = done.loc[window_id] if window_id in done.index else None
    yes_no = [t("yes", lang), t("no", lang)]

    def index_of(column):
        if previous is None or pd.isna(previous[column]):
            return None
        return 0 if int(previous[column]) == 1 else 1

    type_options = ["-"] + TYPES + ["none"]

    def type_label(value):
        if value == "-":
            return t("choose", lang)
        return t("none_type", lang) if value == "none" else f"{value} {TYPE_NAMES[lang][value]}"

    with st.form(f"form_{window_id}"):
        relevant = st.radio(t("q1", lang), yes_no, index=index_of("relevant"), horizontal=True)
        type_ok = st.radio(t("q2", lang), yes_no, index=index_of("type_correct"), horizontal=True)
        start = type_options.index(previous["correct_type"]) if previous is not None and previous["correct_type"] in type_options else 0
        other = st.selectbox(t("q2b", lang), type_options, index=start, format_func=type_label)
        advice_ok = st.radio(t("q3", lang), yes_no, index=index_of("advice_ok"), horizontal=True)
        comment = st.text_area(t("comment", lang), value="" if previous is None else previous["comment"] or "", max_chars=1000)
        if st.form_submit_button(t("save_next", lang), type="primary"):
            if None in (relevant, type_ok, advice_ok):
                st.error(t("need_answers", lang))
            elif type_ok == yes_no[1] and other == "-":
                st.error(t("need_type", lang))
            else:
                review.save_response(folder, user["expert_id"], window_id, relevant == yes_no[0], type_ok == yes_no[0],
                                     None if type_ok == yes_no[0] or other == "-" else other, advice_ok == yes_no[0],
                                     comment, time.time() - st.session_state["timer"])
                remaining = [k for k in open_positions if k != pos]
                later = [k for k in remaining if k > pos]
                st.session_state["position"] = later[0] if later else (remaining[0] if remaining else pos)
                st.rerun()

    b1, b2, b3 = st.columns(3)
    if b1.button(t("previous", lang), disabled=pos == 0):
        st.session_state["position"] = pos - 1
        st.rerun()
    if b2.button(t("next", lang), disabled=pos >= len(order) - 1):
        st.session_state["position"] = pos + 1
        st.rerun()
    if b3.button(t("first_open", lang), disabled=not open_positions):
        st.session_state["position"] = open_positions[0]
        st.rerun()


@st.cache_data(show_spinner=False)
def load_sessions(folder: str):
    folder = Path(folder)
    if not (folder / "session_items.parquet").exists():
        return None
    return (pd.read_parquet(folder / "session_items.parquet"), pd.read_parquet(folder / "session_assignments.parquet"),
            pd.read_parquet(folder / "session_traces.parquet"))


def corner_figure(trace: pd.DataFrame, highlight: int, lang: str) -> go.Figure:
    """All laps of a session in one corner, with the reference lap and one lap highlighted."""
    rows = [("speed", 1.0), ("gas", 100.0), ("brake", 100.0), ("steer", 1.0)]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.34, 0.22, 0.22, 0.22])
    for k, (name, scale) in enumerate(rows, start=1):
        for lap_no, g in trace[trace["lap_no"] > 0].groupby("lap_no"):
            fig.add_trace(go.Scatter(x=g["dist_m"], y=g[name] * scale, showlegend=k == 1 and lap_no == 1, legendgroup="all",
                                     name=t("sess_all_laps", lang), hoverinfo="skip", opacity=0.35,
                                     line=dict(color="#8fa3b8", width=1)), row=k, col=1)
        ref = trace[trace["lap_no"] == 0]
        fig.add_trace(go.Scatter(x=ref["dist_m"], y=ref[name] * scale, showlegend=k == 1, legendgroup="ref",
                                 name=t("reference", lang), line=dict(color=REF_COLOUR, width=2.5, dash="dash")), row=k, col=1)
        lap = trace[trace["lap_no"] == highlight]
        fig.add_trace(go.Scatter(x=lap["dist_m"], y=lap[name] * scale, showlegend=k == 1, legendgroup="lap",
                                 name=f"{t('sess_lap', lang)} {highlight}", line=dict(color=LAP_COLOUR, width=2.5)), row=k, col=1)
        fig.update_yaxes(title_text=t(name, lang), row=k, col=1, title_font_size=11)
    fig.update_xaxes(title_text=t("distance", lang), row=4, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def page_sessions(folder: Path, user: dict, lang: str, budget: int) -> None:
    package = load_sessions(str(folder))
    if package is None:
        st.info(t("dash_empty", lang))
        return
    items, assignments, traces = package
    mine = assignments[assignments["expert_id"] == user["expert_id"]].sort_values("position")
    order = mine["session_key"].tolist()
    if not order:
        st.info(t("dash_empty", lang))
        return
    picks = review.load_session_picks(folder)
    picks = picks[picks["expert_id"] == user["expert_id"]]
    done = set(picks["session_key"])
    open_positions = [k for k, s in enumerate(order) if s not in done]
    if "session_position" not in st.session_state:
        st.session_state["session_position"] = open_positions[0] if open_positions else len(order) - 1
    pos = min(st.session_state["session_position"], len(order) - 1)
    session = order[pos]
    if st.session_state.get("session_timer_for") != session:
        st.session_state["session_timer_for"], st.session_state["session_timer"] = session, time.time()

    st.progress(len(done) / len(order), text=f"{t('sess_progress', lang, i=pos + 1, n=len(order))}   ({len(done)}/{len(order)})")
    if not open_positions:
        st.success(t("sess_done", lang))
    st.write(t("sess_intro", lang, b=budget))
    data = items[items["session_key"] == session]
    corners = data.sort_values("start_idx")[["segment", "segment_name"]].drop_duplicates()
    names = dict(zip(corners["segment"], corners["segment_name"]))
    st.subheader(data["circuit"].iloc[0].replace("_", " ").title())

    grid = data.pivot(index="lap_no", columns="segment", values="total_loss")[corners["segment"].tolist()]
    limit = float(max(grid.abs().quantile(0.97).max(), 0.3))
    heat = go.Figure(go.Heatmap(z=grid.to_numpy(), x=[names[c] for c in grid.columns], y=[str(i) for i in grid.index],
                                colorscale="RdBu_r", zmid=0, zmin=-limit, zmax=limit, colorbar=dict(title="s"),
                                hovertemplate="%{x}, " + t("sess_lap", lang) + " %{y}: %{z:+.2f} s<extra></extra>"))
    heat.update_layout(height=120 + 26 * len(grid), margin=dict(l=10, r=10, t=40, b=10), title=t("sess_heat", lang),
                       yaxis=dict(title=t("sess_lap", lang), autorange="reversed"))
    st.plotly_chart(heat, width="stretch")

    summary = data.groupby("segment").agg(mean=("total_loss", "mean"), spread=("total_loss", "std"),
                                          worst=("total_loss", "max"), off=("tyres_out_max", lambda x: int((x >= 3).sum())))
    summary = summary.loc[corners["segment"]].round(2)
    summary.index = [names[c] for c in summary.index]
    summary.columns = [t("sess_mean", lang), t("sess_spread", lang), t("sess_worst", lang), t("sess_off", lang)]
    with st.expander(t("sess_summary", lang)):
        st.dataframe(summary, width="stretch")

    st.markdown(f"**{t('sess_inspect', lang)}**")
    c1, c2 = st.columns(2)
    corner = c1.selectbox(t("sess_corner", lang), corners["segment"].tolist(), format_func=names.get, key=f"inspect_{session}")
    laps = sorted(data["lap_no"].unique())
    worst = int(data[data["segment"] == corner].sort_values("total_loss").iloc[-1]["lap_no"])
    highlight = c2.selectbox(t("sess_highlight", lang), laps, index=laps.index(worst), key=f"lap_{session}_{corner}")
    st.plotly_chart(corner_figure(traces[(traces["session_key"] == session) & (traces["segment"] == corner)], highlight, lang),
                    width="stretch")

    earlier = picks[(picks["session_key"] == session) & (picks["rank"] > 0)].set_index("rank")
    corner_options = ["-"] + corners["segment"].tolist()
    type_options = ["-"] + TYPES
    with st.form(f"session_{session}"):
        chosen = []
        for k in range(1, budget + 1):
            a, b, c = st.columns([2, 3, 4])
            old = earlier.loc[k] if k in earlier.index else None
            seg = a.selectbox(t("sess_pick", lang, k=k), corner_options,
                              index=corner_options.index(old["segment"]) if old is not None else 0,
                              format_func=lambda v: t("sess_no_corner", lang) if v == "-" else names[v], key=f"seg_{session}_{k}")
            kind = b.selectbox(t("sess_type", lang), type_options,
                               index=type_options.index(old["type"]) if old is not None and old["type"] in type_options else 0,
                               format_func=lambda v: t("choose", lang) if v == "-" else f"{v} {TYPE_NAMES[lang][v]}",
                               key=f"type_{session}_{k}")
            note = c.text_input(t("comment", lang), value="" if old is None else old["comment"] or "", key=f"note_{session}_{k}")
            chosen.append((seg, kind, note))
        if st.form_submit_button(t("save_next", lang), type="primary"):
            selected = [(s_, k_, n_) for s_, k_, n_ in chosen if s_ != "-"]
            if any(k_ == "-" for _, k_, _ in selected):
                st.error(t("sess_need_type", lang))
            elif len({s_ for s_, _, _ in selected}) < len(selected):
                st.error(t("sess_duplicate", lang))
            else:
                review.save_session_picks(folder, user["expert_id"], session,
                                          [dict(segment=s_, type=k_, comment=n_) for s_, k_, n_ in selected],
                                          time.time() - st.session_state["session_timer"])
                later = [k for k in open_positions if k > pos]
                remaining = [k for k in open_positions if k != pos]
                st.session_state["session_position"] = later[0] if later else (remaining[0] if remaining else pos)
                st.rerun()
    b1, b2 = st.columns(2)
    if b1.button(t("previous_session", lang), disabled=pos == 0):
        st.session_state["session_position"] = pos - 1
        st.rerun()
    if b2.button(t("next_session", lang), disabled=pos >= len(order) - 1):
        st.session_state["session_position"] = pos + 1
        st.rerun()


def page_dashboard(folder: Path, user: dict, lang: str) -> None:
    items, assignments, _ = load_package(str(folder))
    responses = review.load_responses(folder)
    st.title(t("page_dash", lang))
    if user["role"] == "admin":
        st.subheader(t("dash_all", lang))
        total = assignments.groupby("expert_id").size().rename("assigned")
        done = responses.groupby("expert_id").agg(reviewed=("window_id", "size"), relevant=("relevant", "mean"),
                                                  median_seconds=("seconds", "median"))
        picks = review.load_session_picks(folder)
        sessions = picks.groupby("expert_id")["session_key"].nunique().rename("sessions_done")
        st.dataframe(pd.concat([total, done, sessions], axis=1).fillna(0).round(2), width="stretch")
        st.download_button(t("dash_download", lang), responses.to_csv(index=False).encode("utf-8"), "responses.csv")
        st.download_button("session_picks.csv", picks.to_csv(index=False).encode("utf-8"), "session_picks.csv")
        return
    mine = responses[responses["expert_id"] == user["expert_id"]]
    n_total = int((assignments["expert_id"] == user["expert_id"]).sum())
    st.subheader(t("dash_mine", lang))
    st.progress(len(mine) / max(n_total, 1), text=f"{t('dash_done', lang)}: {len(mine)} / {n_total}")
    if mine.empty:
        st.info(t("dash_empty", lang))
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("dash_relevant", lang), f"{100 * mine['relevant'].mean():.0f}%")
    c2.metric(t("dash_type_ok", lang), f"{100 * mine['type_correct'].mean():.0f}%")
    c3.metric(t("dash_advice_ok", lang), f"{100 * mine['advice_ok'].mean():.0f}%")
    c4.metric(t("dash_time", lang), f"{mine['seconds'].median():.0f} s")
    joined = mine.merge(items[["window_id", "shown_type"]], on="window_id")
    counts = joined.groupby(["shown_type", "relevant"]).size().unstack(fill_value=0).reindex(TYPES, fill_value=0)
    fig = go.Figure()
    for value, name, colour in ((1, t("yes", lang), LAP_COLOUR), (0, t("no", lang), "#b0b7bf")):
        if value in counts:
            fig.add_bar(x=[f"{k} {TYPE_NAMES[lang][k]}" for k in counts.index], y=counts[value],
                        name=f"{t('dash_relevant', lang)}: {name}", marker_color=colour)
    fig.update_layout(barmode="stack", height=380, title=t("dash_by_type", lang), yaxis_title=t("count", lang),
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Review", layout="wide")
    folder = review_dir()
    with st.sidebar:
        lang = st.radio(T["language"]["en"] + " / " + T["language"]["nl"], ["nl", "en"], horizontal=True,
                        format_func=str.upper, key="lang")
    if not (folder / "items.parquet").exists():
        st.error(t("no_package", lang))
        return
    user = st.session_state.get("user")
    if user is None:
        page_login(folder, lang)
        return
    with st.sidebar:
        pages = ["page_intro", "page_sessions", "page_review", "page_dash"]
        page = st.radio(" ", pages, format_func=lambda k: t(k, lang), label_visibility="collapsed", key="page")
        st.caption(f"{t('logged_in', lang)} {user['expert_id']}")
        if st.button(t("logout", lang)):
            st.session_state.clear()
            st.rerun()
    if page == "page_intro":
        page_intro(folder, user, lang)
    elif page == "page_sessions" and user["role"] == "expert":
        page_sessions(folder, user, lang, int(os.environ.get("KM_BUDGET", load_budget())))
    elif page == "page_review" and user["role"] == "expert":
        page_review(folder, user, lang)
    else:
        page_dashboard(folder, user, lang)


main()
