import streamlit as st
import pandas as pd
import json
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.set_page_config(layout="wide")

# --- Logo and Header ---
st.image("src/pharos_logo.png", width=180)
st.markdown("# Share of Model: Colorado Narrative Intelligence Dashboard")

# --- Load Data Functions ---
@st.cache_data
def load_main():
    df = pd.read_csv("data/looker_export_colorado.csv")
    for col in [
        "issue_topic_affinities", "matched_moral_foundations_semantic_scores",
        "framing_polarity_score", "positive_frame", "negative_frame",
        "framing_role_positive", "framing_role_negative"
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) and isinstance(x, str) and x.startswith("{") else x)
    return df

@st.cache_data
def load_structured():
    try:
        with open("data/enriched_structured_insights.jsonl") as f:
            lines = [json.loads(line) for line in f]
        rows = []
        for r in lines:
            try:
                s = json.loads(r["structured"])
                s.update({
                    "candidate": r["candidate"],
                    "issue": r.get("issue"),
                    "prompt_type": r["prompt_type"]
                })
                rows.append(s)
            except:
                continue
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_framing():
    with open("data/framing_contrast_colorado.jsonl") as f:
        return pd.DataFrame([json.loads(line) for line in f])

@st.cache_data
def load_top_issues():
    with open("data/responses_gpt4_colorado.jsonl") as f:
        return [json.loads(line) for line in f if json.loads(line).get("prompt_type") == "top_issues"]

# --- Load Data ---
df = load_main()
sdf = load_structured()
fdf = load_framing()
top_issues_raw = load_top_issues()

for col in ["salience_score", "salience_mentions", "framing_polarity_score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Tabs ---
tabs = st.tabs(["Roles", "Frames", "Topics", "Salience", "Framing Polarity", "Top Issues", "Narrative Insights"])
candidates = sorted(df["candidate"].dropna().unique())

# --- Tab 0: Roles ---
with tabs[0]:
    st.header("🦸 Narrative Roles")
    selected = st.multiselect("Filter candidates:", candidates, default=candidates)
    filtered = df[df["candidate"].isin(selected)]
    role_counts = filtered.groupby(["candidate", "refined_role_label"]).size().reset_index(name="count")
    chart = alt.Chart(role_counts).mark_bar().encode(
        x=alt.X("candidate:N", sort="-y"),
        y="count:Q",
        color=alt.Color("refined_role_label:N", scale=alt.Scale(domain=["Hero", "Neutral", "Villain"], range=["#00C853", "#B0BEC5", "#E53935"])),
        tooltip=["candidate", "refined_role_label", "count"]
    )
    st.altair_chart(chart, use_container_width=True)

# --- Tab 1: Frames ---

def safe_normalize(g):
    total = g["score"].sum()
    if total == 0 or pd.isna(total):
        g["norm_score"] = 0
    else:
        g["norm_score"] = (g["score"] / total) * 100
    return g

with tabs[1]:
    st.header("🧠 Semantic Frames")
    frame_rows = []
    for _, row in df.iterrows():
        try:
            parsed = row.get("matched_frames_semantic_scores", {})
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            for frame, score in parsed.items():
                frame_rows.append({"candidate": row["candidate"], "frame": frame, "score": float(score)})
        except:
            continue
    frame_df = pd.DataFrame(frame_rows)
    if not frame_df.empty:
        norm = frame_df.groupby("candidate").apply(safe_normalize).reset_index(drop=True)
        chart = alt.Chart(norm).mark_rect().encode(
            x="frame:N", y="candidate:N",
            color=alt.Color("norm_score:Q", scale=alt.Scale(scheme="greens")),
            tooltip=["candidate", "frame", "norm_score"]
        )
        st.altair_chart(chart, use_container_width=True)

# --- Tab 2: Topics ---
with tabs[2]:
    st.header("🧭 Issue Topic Affinities")
    topic_rows = []
    for _, row in df.iterrows():
        for topic_obj, score in row.get("issue_topic_affinities", {}).items():
            if isinstance(topic_obj, dict):
                topic_label = topic_obj.get("label") or json.dumps(topic_obj)
            elif isinstance(topic_obj, (list, tuple)):
                topic_label = ", ".join(map(str, topic_obj))
            else:
                topic_label = str(topic_obj)

            topic_rows.append({
                "candidate": row["candidate"],
                "topic": topic_label.strip(),
                "score": score
            })
    topic_df = pd.DataFrame(topic_rows)
    st.write("Topic rows:", len(topic_rows))
    st.write("Unique topics:", topic_df["topic"].nunique())
    if not topic_df.empty:
        pivot = topic_df.groupby(["candidate", "topic"])["score"].mean().reset_index()
        pivot["topic"] = pivot["topic"].astype(str)  # force string again
        chart = alt.Chart(pivot).mark_rect().encode(
            x=alt.X(
                "topic:N",
                sort="-y",
                axis=alt.Axis(
                    labelAngle=-45,
                    labelLimit=999,         # allow full width
                    labelOverlap=False,     # don't drop labels
                    labelFlush=False,
                    labelExpr="datum"       # show full label
                )
            ),
            y="candidate:N",
            color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["candidate", "topic", "score"]
        )
        st.altair_chart(chart, use_container_width=True)

# --- Tab 3: Salience ---
with tabs[3]:
    st.header("📢 Salience Overview")
    salient = df[df["salience_score"].notnull() & df["salience_mentions"].notnull()]
    if not salient.empty:
        scatter = alt.Chart(salient).mark_circle(size=150).encode(
            x="salience_mentions:Q", y="salience_score:Q",
            color="candidate:N",
            tooltip=["candidate", "issue", "salience_score", "salience_mentions"]
        )
        st.altair_chart(scatter, use_container_width=True)

# --- Tab 4: Framing Polarity ---
with tabs[4]:
    st.header("🪞 Framing Polarity")
    framing_df = fdf[fdf["prompt_type"] == "framing"]
    framing_df["framing_polarity_score"] = pd.to_numeric(framing_df["framing_polarity_score"], errors="coerce")
    framing_df = framing_df[framing_df["framing_polarity_score"].notnull()]

    st.subheader("Candidate × Issue Matrix")
    chart = alt.Chart(framing_df).mark_circle(size=250).encode(
        x="issue:N", y="candidate:N",
        color=alt.Color("framing_polarity_score:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#FF0051", "#F0F0F0", "#00FF00"])),
        tooltip=["candidate", "issue", "framing_polarity_score"]
    )
    st.altair_chart(chart, use_container_width=True)

# --- Tab 5: Top Issues ---
with tabs[5]:
    st.header("📋 Top Issues by Candidate")
    issue_map = {}
    for r in top_issues_raw:
        lines = r["response"].split("\n")
        extracted = [line.split(":")[0].strip() for line in lines if line.strip() and line[0].isdigit()]
        issue_map[r["candidate"]] = extracted[:10]

    selected = st.multiselect("Select candidates", sorted(issue_map.keys()), default=list(issue_map.keys())[:4])
    cols = st.columns(max(len(selected), 1))
    for i, cand in enumerate(selected):
        cols[i].markdown(f"**{cand}**")
        for j, issue in enumerate(issue_map.get(cand, []), 1):
            cleaned = re.sub(r"^\d+\.\s*", "", issue)  # correct regex
            cleaned = re.sub(r"^\d+\.\s*", "", issue).strip()
            cols[i].markdown(f"{j}. {cleaned}")

# --- Tab 6: Structured Narrative Insights ---
with tabs[6]:
    st.header("🔍 Narrative Insights (LLM Extracted)")

    if sdf.empty:
        st.warning("Structured insights not loaded — waiting for data.")
    else:
        pt = st.selectbox("Prompt Type", sorted(sdf["prompt_type"].unique()))
        cand = st.selectbox("Candidate", sorted(sdf["candidate"].unique()), key="structured_cand")
        sub = sdf[(sdf["prompt_type"] == pt) & (sdf["candidate"] == cand)]

        if pt == "voter_pov":
            for _, row in sub.iterrows():
                st.markdown(f"**Issue: {row['issue']}**")
                st.markdown("- Strengths: " + ", ".join(row.get("strengths", [])))
                st.markdown("- Concerns: " + ", ".join(row.get("concerns", [])))
                st.markdown(f"**Persuasiveness Score:** {row.get('persuasiveness_score', 'N/A')}")
                st.divider()
        elif pt == "moral":
            tokens = []
            for _, row in sub.iterrows():
                tokens.extend(row.get("moral_values", []))
            if tokens:
                wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(tokens))
                st.image(wc.to_image())
            else:
                st.info("No moral values found.")
        elif pt == "salience":
            for _, row in sub.iterrows():
                st.markdown(f"**Issue: {row['issue']}**")
                st.markdown("- Top Known Policies: " + ", ".join(row.get("top_known_policies", [])))
                st.markdown("- Misconceptions: " + ", ".join(row.get("misconceptions", [])))
                st.markdown(f"**Visibility Score:** {row.get('media_visibility_score', 'N/A')}")
                st.divider()
        elif pt == "contrastive":
            for _, row in sub.iterrows():
                st.markdown(f"**Issue: {row['issue']}**")
                st.markdown("- Aligned With Party On: " + ", ".join(row.get("aligned_with_party_on", [])))
                st.markdown("- Differs From Party On: " + ", ".join(row.get("differs_from_party_on", [])))
                st.markdown(f"**Contrast Score:** {row.get('contrast_score', 'N/A')}")
                st.divider()
