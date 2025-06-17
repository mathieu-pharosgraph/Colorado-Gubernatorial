import streamlit as st
import pandas as pd
import json
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from pathlib import Path

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
    path = Path("data/enriched_structured_insights.jsonl")
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                structured_raw = r.get("structured", "").strip()
                if structured_raw.startswith("```json"):
                    structured_raw = structured_raw.replace("```json", "").replace("```", "").strip()
                s = json.loads(structured_raw)
                s.update({
                    "candidate": r["candidate"],
                    "issue": r.get("issue"),
                    "prompt_type": r["prompt_type"]
                })
                rows.append(s)
            except Exception as e:
                continue  # optional: log error if needed
    return pd.DataFrame(rows)


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
tabs = st.tabs([
    "Roles", 
    "Framing Polarity", "Top Issues", "Framing Wordclouds", "Narrative Insights", "Voter Persuasion Insights", "Frames", "Topics"
])

candidates = sorted(df["candidate"].dropna().unique())

# --- Tab 0: Roles ---
with tabs[0]:
    st.header("🦸 Narrative Roles")
    selected = st.multiselect("Filter candidates:", candidates, default=candidates)
    filtered = df[df["candidate"].isin(selected)]
    role_counts = filtered.groupby(["candidate", "refined_role_label"]).size().reset_index(name="count")
    total_counts = role_counts.groupby("candidate")["count"].transform("sum")
    role_counts["percentage"] = (role_counts["count"] / total_counts * 100).round(1)

    chart = alt.Chart(role_counts).mark_bar().encode(
        x=alt.X("candidate:N", sort="-y"),
        y="count:Q",
        color=alt.Color("refined_role_label:N", scale=alt.Scale(
            domain=["Hero", "Neutral", "Villain"],
            range=["#00C853", "#B0BEC5", "#E53935"]
        )),
        tooltip=[
            alt.Tooltip("candidate:N"),
            alt.Tooltip("refined_role_label:N", title="Role"),
            alt.Tooltip("count:Q", title="Count"),
            alt.Tooltip("percentage:Q", title="% of Candidate", format=".1f")
        ]
    )
    st.altair_chart(chart, use_container_width=True)






# --- Tab 1: Framing Polarity ---
with tabs[1]:
    st.header("🪞 Framing Polarity")
    framing_df = fdf[fdf["prompt_type"] == "framing"]
    framing_df["framing_polarity_score"] = pd.to_numeric(framing_df["framing_polarity_score"], errors="coerce")
    framing_df = framing_df[framing_df["framing_polarity_score"].notnull()]

    st.subheader("Candidate × Issue Matrix")
    chart = alt.Chart(framing_df).mark_circle(size=250).encode(
        x="issue:N", y="candidate:N",
        color=alt.Color("framing_polarity_score:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#FF0051", "#F0F0F0", "#00FF00"])),
        tooltip=[
            alt.Tooltip("candidate:N"),
            alt.Tooltip("issue:N"),
            alt.Tooltip("framing_polarity_score:Q", title="Polarity Score", format=".2f")
        ]
    )
    st.altair_chart(chart, use_container_width=True)

# --- Tab 2: Top Issues ---
with tabs[2]:
    st.header("📋 Top Issues by Candidate")

    issue_map = {}
    for r in top_issues_raw:
        candidate = r["candidate"]
        matches = re.findall(r"\d+\.\s+(.*)", r["response"])
        cleaned = [m.strip() for m in matches if m.strip()]
        if cleaned:
            issue_map[candidate] = cleaned[:10]


    selected = st.multiselect("Select candidates", sorted(issue_map.keys()), default=list(issue_map.keys())[:4])
    cols = st.columns(max(len(selected), 1))
    for i, cand in enumerate(selected):
        cols[i].markdown(f"**{cand}**")
        for j, issue in enumerate(issue_map.get(cand, []), 1):
            cleaned = re.sub(r"^\d+\.\s*", "", issue)  # correct regex
            cleaned = re.sub(r"^\d+\.\s*", "", issue).strip()
            cols[i].markdown(f"{j}. {cleaned}")

# --- Tab 3: Framing Wordclouds ---
with tabs[3]:
    st.header("🧩 Narrative Tags & Wordclouds")

    @st.cache_data
    def load_narratives():
        with open("data/framing_narratives_enriched_colorado.jsonl") as f:
            return pd.DataFrame([json.loads(line) for line in f if json.loads(line).get("prompt_type") == "framing"])

    narratives = load_narratives()
    candidate_filter = st.selectbox("Select a candidate:", sorted(narratives["candidate"].dropna().unique()), key="distinct_candidate")
    issue_filter = st.multiselect("Select issue(s):", sorted(narratives["issue"].dropna().unique()), default=narratives["issue"].dropna().unique(), key="distinct_issues")

    filtered = narratives[
        (narratives["candidate"] == candidate_filter) &
        (narratives["issue"].isin(issue_filter))
    ]

    def display_wordcloud(title, field):
        st.subheader(title)
        tokens = []
        for items in filtered[field].dropna():
            if isinstance(items, list):
                tokens.extend(items)
        if tokens:
            text = " ".join(tokens)
            wc = WordCloud(width=800, height=300, background_color="white").generate(text)
            st.image(wc.to_image())
        else:
            st.info("No data to display.")

    # Wordclouds to show
    display_wordcloud("Traits", "traits")
    display_wordcloud("Themes", "themes")
    display_wordcloud("Criticisms", "criticisms")
    display_wordcloud("Archetypes", "archetypes")
    display_wordcloud("Alignments", "alignments")

# --- Tab 4: Structured Narrative Insights ---
with tabs[4]:
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

# --- Tab X: Voter Persuasion Insights ---
with tabs[5]:  # adjust index if needed
    st.header("🧠 Voter Persuasion Insights")

    # Filter structured voter_pov entries
    voter_df = sdf[sdf["prompt_type"] == "voter_pov"]
    if voter_df.empty:
        st.info("No voter perception data available.")
    else:
        st.subheader("🎯 Persuasiveness Score by Candidate & Issue")
        voter_df["persuasiveness_score"] = pd.to_numeric(voter_df["persuasiveness_score"], errors="coerce")
        pscore_chart = alt.Chart(voter_df.dropna(subset=["persuasiveness_score"])).mark_circle(size=150).encode(
            x=alt.X("issue:N", title="Issue", sort="ascending"),
            y=alt.Y("candidate:N", title="Candidate"),
            color=alt.Color("persuasiveness_score:Q", scale=alt.Scale(scheme="greens")),
            size="persuasiveness_score:Q",
            tooltip=["candidate", "issue", alt.Tooltip("persuasiveness_score:Q", format=".2f")]
        ).properties(height=400)
        st.altair_chart(pscore_chart, use_container_width=True)

        st.subheader("📏 Issue Clarity by Candidate & Issue")
        voter_df["issue_clarity"] = pd.to_numeric(voter_df["issue_clarity"], errors="coerce")
        clarity_chart = alt.Chart(voter_df.dropna(subset=["issue_clarity"])).mark_circle(size=150).encode(
            x=alt.X("issue:N", title="Issue", sort="ascending"),
            y=alt.Y("candidate:N", title="Candidate"),
            color=alt.Color("issue_clarity:Q", scale=alt.Scale(scheme="blues")),
            size="issue_clarity:Q",
            tooltip=["candidate", "issue", alt.Tooltip("issue_clarity:Q", format=".1f")]
        ).properties(height=400)
        st.altair_chart(clarity_chart, use_container_width=True)

        st.subheader("👥 Demographic Appeal Word Cloud")
        selected_candidate = st.selectbox("Select candidate", sorted(voter_df["candidate"].dropna().unique()), key="demo_wc_cand")
        issues_for_cand = voter_df[voter_df["candidate"] == selected_candidate]["issue"].dropna().unique().tolist()
        selected_issues = st.multiselect("Optional: Filter by issue(s)", issues_for_cand, default=issues_for_cand)

        demo_df = voter_df[
            (voter_df["candidate"] == selected_candidate) &
            (voter_df["issue"].isin(selected_issues))
        ]

        tokens = []
        for appeal in demo_df["demographic_appeal"].dropna():
            if isinstance(appeal, list):
                tokens.extend(appeal)

        if tokens:
            wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(tokens))
            st.image(wc.to_image())
        else:
            st.info("No demographic appeal data available.")

# --- Tab 6: Frames ---
with tabs[6]:
    st.header("🧠 Semantic Frames")
    frame_rows = []
    for _, row in df.iterrows():
        parsed = row.get("matched_frames_semantic_scores", {})
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        for frame, score in parsed.items():
            frame_rows.append({"candidate": row["candidate"], "frame": frame, "score": float(score)})
    frame_df = pd.DataFrame(frame_rows)

    def safe_normalize(g):
        total = g["score"].sum()
        g["score"] = (g["score"] / total * 100) if total > 0 else 0
        return g

    if not frame_df.empty:
        norm = frame_df.groupby("candidate", group_keys=False).apply(safe_normalize).rename(columns={"score": "norm_score"})
        chart = alt.Chart(norm).mark_rect().encode(
            x=alt.X("frame:N"),
            y=alt.Y("candidate:N"),
            color=alt.Color("norm_score:Q", scale=alt.Scale(scheme="greens")),
            tooltip=[
                alt.Tooltip("candidate:N"),
                alt.Tooltip("frame:N"),
                alt.Tooltip("norm_score:Q", title="Score", format=".2f")
            ]
        )
        st.altair_chart(chart, use_container_width=True)



# --- Tab 7: Topics ---
with tabs[7]:
    st.header("🧭 Issue Topic Affinities")
    topic_rows = []
    for _, row in df.iterrows():
        for topic_obj, score in row.get("issue_topic_affinities", {}).items():
            try:
                if isinstance(topic_obj, dict):
                    topic_label = topic_obj.get("label") or topic_obj.get("name") or str(topic_obj)
                elif isinstance(topic_obj, (list, tuple)):
                    topic_label = ", ".join(map(str, topic_obj))
                else:
                    topic_label = str(topic_obj)
            except Exception:
                topic_label = "Unlabeled Topic"

            topic_rows.append({
                "candidate": row["candidate"],
                "topic": topic_label.strip(),
                "score": score
            })


    topic_df = pd.DataFrame(topic_rows)
    if not topic_df.empty:
        pivot = topic_df.groupby(["candidate", "topic"])["score"].mean().reset_index()
        pivot = pivot[pivot["score"].notnull() & ~pivot["score"].isin([float("inf"), float("-inf")])]
        pivot["topic"] = pivot["topic"].astype(str)

        chart = alt.Chart(pivot).mark_rect().encode(
            x=alt.X("topic:N", sort="-y", axis=alt.Axis(labelAngle=-45, labelLimit=0,labelOverlap=False)),
            y="candidate:N",
            color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["candidate", "topic", "score"]
        )
        st.altair_chart(chart, use_container_width=True)