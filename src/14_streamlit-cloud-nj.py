import streamlit as st
import pandas as pd
import json
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from pathlib import Path
import geopandas as gpd
import pydeck as pdk
from hashlib import md5
import os
import warnings
import time
import ast

def login():
    st.title("🔐 Login")
    password = st.text_input("Enter password", type="password")
    if password == st.secrets["app_password_nj"]:
        st.session_state["logged_in"] = True
    else:
        st.stop()

if "logged_in" not in st.session_state:
    login()

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

st.set_page_config(layout="wide")

# --- Logo and Header ---
st.image("src/pharos_logo.png", width=180)
st.markdown("# Model Share: New Jersey Narrative Intelligence Dashboard")

# --- Load Data Functions ---
@st.cache_data
def load_main():
    df = pd.read_csv("data/new_jersey/looker_export_nj.csv")
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
    path = Path("data/new_jersey/enriched_structured_insights_nj.jsonl")
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
                continue
    return pd.DataFrame(rows)

@st.cache_data
def load_framing():
    with open("data/new_jersey/framing_contrast_nj.jsonl") as f:
        return pd.DataFrame([json.loads(line) for line in f])

@st.cache_data
def load_top_issues():
    with open("data/new_jersey/responses_gpt4_nj.jsonl") as f:
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
    "Framing Polarity", "Top Issues", "Framing Wordclouds", "Narrative Insights", "Voter Persuasion Insights", "State Tables", "Census Block Group Maps", "Census Block Group Tables", "Opposition Insights", "Frames", "Topics"
])

candidates = sorted(df["candidate"].dropna().unique())

@st.cache_data
def load_blockgroup_data():
    shapes = gpd.read_file("data/new_jersey/tl_2020_34_bg.shp", engine="fiona")

    fips_to_name = {
        "001": "atlantic",
        "003": "bergen",
        "005": "burlington",
        "007": "camden",
        "009": "cape may",
        "011": "cumberland",
        "013": "essex",
        "015": "gloucester",
        "017": "hudson",
        "019": "hunterdon",
        "021": "mercer",
        "023": "middlesex",
        "025": "monmouth",
        "027": "morris",
        "029": "ocean",
        "031": "passaic",
        "033": "salem",
        "035": "somerset",
        "037": "sussex",
        "039": "union",
        "041": "warren"
    }

    shapes["county_fips"] = shapes["COUNTYFP"].astype(str).str.zfill(3)
    shapes["county_name"] = shapes["county_fips"].map(fips_to_name)
    shapes["block_group"] = shapes["GEOID"].astype(str)

    shapes["geometry"] = shapes["geometry"].simplify(0.001, preserve_topology=True)

    shapes = shapes[["county_name", "block_group", "geometry"]]

    scores = pd.read_csv("data/new_jersey/precinct_candidate_scores_with_confidence_nj.csv")
    scores["block_group"] = scores["precinct_num"].astype(str).str.zfill(3)
    scores["county_name"] = scores["county_name"].str.strip().str.lower()
    scores["block_group"] = scores["block_group"]

    turnout = pd.read_csv("data/new_jersey/block_level_census_voting.csv")
    turnout["block_group"] = turnout["block_group"].astype(str).str.zfill(3)
    turnout["county_name"] = turnout["county"].str.strip().str.lower()
    turnout["block_group"] = turnout["block_group"]

    merged = shapes.merge(scores, on=["county_name", "block_group"], how="inner")
    merged = merged.merge(turnout[["county_name", "block_group", "totalvotes", "turnout"]],
                          on=["county_name", "block_group"], how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=shapes.crs), shapes



gdf, shapes = load_blockgroup_data()



def name_to_rgb(name):
    h = md5(name.encode()).hexdigest()
    return [int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16), 200]

@st.cache_data
def to_geojson_cached(_df, key: str):
    return json.loads(_df.to_json())

def show_pydeck_map(gdf_map, value_col, candidate_name=None, other_score_col=None, second_name=None):
    if gdf_map.empty:
        st.info("No data to show on map.")
        return

    gdf_map = gdf_map.dropna(subset=["geometry"]).copy()

    # ✅ Only round if the column is numeric
    if pd.api.types.is_numeric_dtype(gdf_map[value_col]):
        gdf_map[value_col] = gdf_map[value_col].round(2)

    if "rgb" not in gdf_map.columns:
        if pd.api.types.is_numeric_dtype(gdf_map[value_col]):
            gdf_map["score_norm"] = (gdf_map[value_col] - gdf_map[value_col].min()) / (
                gdf_map[value_col].max() - gdf_map[value_col].min() + 1e-6
            )
            gdf_map["rgb"] = gdf_map["score_norm"].apply(lambda x: [int(255 * (1 - x)), int(255 * x), 50, 200])
        else:
            gdf_map["rgb"] = gdf_map[value_col].apply(name_to_rgb)

    gdf_map[["r", "g", "b", "a"]] = pd.DataFrame(gdf_map["rgb"].tolist(), index=gdf_map.index)
    gdf_map = gdf_map.to_crs(epsg=4326)

    geojson = to_geojson_cached(gdf_map, key=f"{value_col}_{candidate_name}_{gdf_map.shape[0]}")

    tooltip = f"""
        <b>Candidate:</b> {candidate_name if candidate_name else 'N/A'}<br>
        <b>County:</b> {{county_name}}<br>
        <b>Precinct:</b> {{block_group}}<br>
        <b>{value_col.title()}:</b> {{{value_col}}}
    """
    if value_col == "net_score" and second_name:
        tooltip = f"""
            <b>{candidate_name}:</b> {{cand1_score}}<br>
            <b>{second_name}:</b> {{cand2_score}}<br>
            <b>Net Score:</b> {{net_score}}<br>
            <b>County:</b> {{county_name}}<br>
            <b>Precinct:</b> {{block_group}}
        """

    view_state = pdk.ViewState(latitude=39.0, longitude=-105.5, zoom=7.5)
    layer = pdk.Layer("GeoJsonLayer", data=geojson,
                      get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                      get_line_color=[0, 0, 0, 160], pickable=True, auto_highlight=True)

    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state,
                             layers=[layer], tooltip={"html": tooltip, "style": {"backgroundColor": "black", "color": "white"}}))



# --- Tab 0: Roles ---
with tabs[0]:
    st.header("🦸 Narrative Roles")
    selected = st.multiselect("Filter candidates:", candidates, default=candidates)
    filtered = df[df["candidate"].isin(selected)]
    with st.expander("ℹ️ What does this chart show?"):
        st.markdown("""
        This chart shows the **distribution of narrative roles** for each candidate, as assigned by the AI.

        - **Hero**: Framed positively, with agency and alignment to valued goals.
        - **Villain**: Framed negatively, responsible for harm or conflict.
        - **Neutral**: Described without clear moral or narrative positioning.

        The percentages reflect **how often each candidate was framed in that role** across the underlying media content.
        """)

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
    with st.expander("ℹ️ What does the framing polarity chart show?"):
        st.markdown("""
        This matrix shows how **positively or negatively** each candidate is framed on different issues.

        - **Green = Positive framing** (e.g., praised, aligned with solution)
        - **Red = Negative framing** (e.g., blamed, criticized, associated with harm)
        - **Gray = Neutral framing**

        The polarity score ranges from **-1 (highly negative)** to **+1 (highly positive)** and is based on AI analysis of narrative tone and implications.
        """)

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
    with st.expander("ℹ️ What does the Top Issues view show?"):
        st.markdown("""
        This view lists the **top priority issues** for each candidate, based on how they present themselves to voters.

        - These are extracted from AI-analyzed responses to prompts like:
        *"What are the top 10 issues this candidate is focused on?"*
        - The ordering reflects **how prominently** each issue appears in the candidate's communication.
        
        Use this view to compare **issue priorities** across candidates and see where their agendas align or diverge.
        """)

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
    with st.expander("ℹ️ What do the Narrative Tags and Wordclouds show?"):
        st.markdown("""
        This view displays **common narrative elements** used to describe each candidate on key issues.

        - Tags include:
        - **Traits** (e.g., "decisive", "corrupt")
        - **Themes** (e.g., "economic freedom", "social justice")
        - **Criticisms**, **Archetypes**, and **Alignments**
        - These were extracted using LLMs from how the candidate is portrayed in public discourse.

        The word clouds give a **visual summary** of recurring narrative language—larger words appear more often.

        Use this view to explore how **framing, praise, and critique** vary across issues and candidates.
        """)

    st.header("🧩 Narrative Tags & Wordclouds")

    @st.cache_data
    def load_narratives():
        with open("data/new_jersey/framing_narratives_enriched_nj.jsonl") as f:
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
    st.header("🔍 Narrative Insights")
    with st.expander("ℹ️ What are Structured Narrative Insights?"):
        st.markdown("""
        This view summarizes how each **candidate frames key issues** using structured LLM prompts. Each cell reflects a **short narrative extract** generated by the model, aligned with four lenses:

        - **Contrastive**: How the candidate differentiates their position from others in their party.
        - **Moral**: The moral values or ethical language used (e.g. fairness, harm, liberty).
        - **Salience**: What is the candidate best known for in the context of an issue and what misconceptions exist.
        - **Voter POV**: Strong/weak points for the candidate from the **perspective of swing voters**.

        Use this to compare how candidates not only state their views, but construct **meaningful narratives** around them.
        """)

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
    with st.expander("ℹ️ How to interpret Persuasion and Clarity scores"):
        st.markdown("""
        This view summarizes how **persuasive** and **clear** each candidate's responses appear to voters.

        - **Persuasion Score**: Reflects how compelling and emotionally resonant the candidate’s arguments are.
        - High scores often reflect strong framing, vivid language, or motivational tone.
        - **Clarity Score**: Measures how easy the language is to follow and how clearly the core point is communicated.
        - High clarity often reflects concise sentences, minimal jargon, and a clear logical structure.
        
        **Use Case**: Spot candidates who may be inspiring but vague, or others who are clear but less emotionally persuasive. Look for balance in both dimensions.
        """)


    # Filter structured voter_pov entries
    voter_df = sdf[sdf["prompt_type"] == "voter_pov"]
    if voter_df.empty:
        st.info("No voter perception data available.")
    else:
        st.subheader("🎯 Persuasiveness Score by Candidate & Issue")
        voter_df["persuasiveness_score"] = pd.to_numeric(voter_df["persuasiveness_score"], errors="coerce")
        pscore_chart = alt.Chart(voter_df.dropna(subset=["persuasiveness_score"])).mark_circle(size=150).encode(
            x=alt.X("candidate:N", title="Candidate"),
            y=alt.Y("issue:N", title="Issue", sort="ascending"),
            color=alt.Color("persuasiveness_score:Q", scale=alt.Scale(scheme="greens")),
            size="persuasiveness_score:Q",
            tooltip=["candidate", "issue", alt.Tooltip("persuasiveness_score:Q", format=".2f")]
        ).properties(height=400)
        st.altair_chart(pscore_chart, use_container_width=True)

        st.subheader("📏 Issue Clarity by Candidate & Issue")
        voter_df["issue_clarity"] = pd.to_numeric(voter_df["issue_clarity"], errors="coerce")
        clarity_chart = alt.Chart(voter_df.dropna(subset=["issue_clarity"])).mark_circle(size=150).encode(
            x=alt.X("candidate:N", title="Candidate"),
            y=alt.Y("issue:N", title="Issue", sort="ascending"),
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

with tabs[6]:
    st.header("📈 State Narrative Scores and Head-to-Head Matchups")

    with st.expander("ℹ️ What do the state-level scores mean?"):
        st.markdown("""
        **📊 State Score** – Weighted average narrative strength of each candidate, where each Census Block Group contributes:  
        **Raw Score × Number of Voters × Turnout**  
        normalized by total **Voters × Turnout** across Census Blocks with scores.
        Note: Turnout at the Census Block Group level is a proxy calculated by dividing the number of ballots cast by total population in the block group.

        This reflects **narrative alignment scaled by potential electoral impact**.
        """)

    # Calculate state-level weighted scores
    gdf_valid = gdf[gdf["score"].notnull()]
    gdf_valid["weight"] = gdf_valid["totalvotes"] * gdf_valid["turnout"]

    state_scores = gdf_valid.groupby("candidate").apply(
        lambda g: (g["score"] * g["weight"]).sum() / g["weight"].sum()
    ).round(2).reset_index(name="State Score")

    st.subheader("🏅 State Score Rankings")
    st.dataframe(state_scores.sort_values("State Score", ascending=False), use_container_width=True)

    # Head-to-head filters
    st.subheader("🤜 Head-to-Head Analysis")
    col1, col2 = st.columns(2)
    cand1 = col1.selectbox("Candidate A", candidates, index=candidates.index("Jack Ciattarelli"))
    cand2 = col2.selectbox("Candidate B", candidates, index=candidates.index("Mikie Sherrill"))

    if cand1 != cand2:
        pivot = gdf[gdf["candidate"].isin([cand1, cand2])].pivot_table(
            index=["county_name", "block_group", "totalvoters", "totalvoterturnout1"],
            columns="candidate", values="score"
        ).dropna().reset_index()

        # Compute protection and opportunity scores
        pivot["weight"] = pivot["totalvoters"] * pivot["totalvoterturnout1"]
        margin = (pivot[cand1] - pivot[cand2]).clip(lower=0.01)  # avoid divide-by-zero
        pivot["protect_score"] = ((pivot[cand1] > pivot[cand2]).astype(int)) * (pivot["weight"] / (1 + margin))
        pivot["opportunity_score"] = ((pivot[cand1] < pivot[cand2]).astype(int)) * (1 / (1 + (pivot[cand2] - pivot[cand1]).abs())) * pivot["weight"]

        # Protect table
        st.subheader(f"🛡️ Census Blocks to Protect for {cand1}")
        with st.expander("ℹ️ What do the Census Block Groups to protect scores mean?"):
            st.markdown("""
            **📊 Protect Score** – Considers only the Census Block Groups where the first candidate LEADS the second candidates. Score then based on number of voters * turnout weighted by the inverse of how much the first candidate leads in that block (i.e., Blocks where the first candidate leads by only a small margin will show higher scores (for the same voters * turnout)). 

            """)
        protect_table = pivot[pivot["protect_score"] > 0][[
            "county_name", "block_group", cand1, cand2, "protect_score"
        ]].rename(columns={
            cand1: f"{cand1} Score",
            cand2: f"{cand2} Score"
        }).sort_values("protect_score", ascending=False)
        st.dataframe(protect_table, use_container_width=True)

        # Opportunity table
        st.subheader(f"🚀 Opportunity Census Block Groups for {cand1}")
        with st.expander("ℹ️ What do the opportinity Census Block scores mean?"):
            st.markdown("""
            **📊 Opportunity Score** – Considers only the Census Blocks where the first candidate LAGS the second candidates. Score then based on number of voters * turnout weighted by the inverse of how much the first candidate lags in that block (i.e., blocks where the first candidate lags by only a small margin will show higher scores (for the same voters * turnout)). 

            """)
        opportunity_table = pivot[pivot["opportunity_score"] > 0][[
            "county_name", "block_group", cand1, cand2, "opportunity_score"
        ]].rename(columns={
            cand1: f"{cand1} Score",
            cand2: f"{cand2} Score"
        }).sort_values("opportunity_score", ascending=False)
        st.dataframe(opportunity_table, use_container_width=True)


with tabs[7]:
    with st.expander("ℹ️ What do the scores mean?"):
        st.markdown("""
    **🧠 Raw Score** – AI-detected narrative alignment in this Census Block Group for the selected candidate  
    **📉 Normalized Score** – Relative strength of support (0–100) within the candidate’s Census Block Group footprint  
    **⚖️ Net Score** – The difference in narrative strength between two candidates in a given Census Block Group 
    **🎯 Priority Score** – Weighted opportunity metric based on:  
    **Normalized Score × Voter Base × Turnout**

    > These scores are based on AI analysis of how each candidate is framed in the news — as a **hero or villain**, with specific **traits, archetypes, issue framing**, and alignment with **socio-demographic and political preferences**.
    """)
    st.header("🗺️ Precinct Score Maps")
    candidates = sorted(gdf["candidate"].dropna().unique())
    cand1 = st.selectbox("Map Candidate A", ["All"] + candidates, index=candidates.index("Jack Ciattarelli") + 1)
    cand2 = st.selectbox("Map Candidate B", ["All"] + candidates, index=candidates.index("Mikie Sherrill") + 1)
    filtered = gdf.copy()

    # Pivot table with all candidate scores for winner map later
    pivot_all = gdf.pivot(index=["county_name", "block_group"], columns="candidate", values="score").reset_index()

    # --- Map for Candidate A ---
    if cand1 != "All":
        st.subheader(f"🗺️ Precinct Scores for {cand1}")
        pivot1 = gdf[gdf["candidate"] == cand1][["county_name", "block_group", "score"]].copy()
        pivot1 = pivot1.rename(columns={"score": "cand1_score"})
        df1 = shapes.copy().merge(pivot1, on=["county_name", "block_group"], how="inner")
        df1 = df1.rename(columns={"cand1_score": "score"})
        show_pydeck_map(df1, "score", candidate_name=cand1)

    # --- Map for Candidate B ---
    if cand2 != "All":
        st.subheader(f"🗺️ Precinct Scores for {cand2}")
        pivot2 = gdf[gdf["candidate"] == cand2][["county_name", "block_group", "score"]].copy()
        pivot2 = pivot2.rename(columns={"score": "cand2_score"})
        df2 = shapes.copy().merge(pivot2, on=["county_name", "block_group"], how="inner")
        df2 = df2.rename(columns={"cand2_score": "score"})
        show_pydeck_map(df2, "score", candidate_name=cand2)





    if cand1 != "All" and cand2 != "All" and cand1 != cand2:
        wide = filtered.pivot(index=["county_name", "block_group"], columns="candidate", values="score").reset_index()
        wide = wide.dropna(subset=[cand1, cand2])
        wide["net_score"] = (wide[cand1] - wide[cand2]).round(2)
        wide["cand1_score"] = wide[cand1].round(2)
        wide["cand2_score"] = wide[cand2].round(2)
        geo_net = shapes[["county_name", "block_group", "geometry"]].merge(
            wide[["county_name", "block_group", "net_score", "cand1_score", "cand2_score"]],
            on=["county_name", "block_group"],
            how="inner"
        )

        st.subheader(f"Net Support: {cand1} – {cand2}")
        show_pydeck_map(geo_net, "net_score", candidate_name=cand1, second_name=cand2)

    # --- 🏆 Winner Map ---
    st.subheader("🏆 Winner Map (Top Score)")


    winner_df = gdf.copy()
    pivot = winner_df.pivot_table(index=["county_name", "block_group"], columns="candidate", values="score")
    score_only = pivot[candidates]  # restrict to candidate score columns
    pivot["winner"] = score_only.idxmax(axis=1)
    pivot["max_score"] = score_only.max(axis=1).round(2)


    winner_geo = shapes.merge(pivot.reset_index(), on=["county_name", "block_group"], how="inner")
    winner_geo["rgb"] = winner_geo["winner"].apply(name_to_rgb)
    winner_geo[["r", "g", "b", "a"]] = pd.DataFrame(winner_geo["rgb"].tolist(), index=winner_geo.index)

    show_pydeck_map(winner_geo, "max_score", candidate_name="Winner")

    st.markdown("### 🎨 Winner Legend")
    for cand in sorted(winner_geo["winner"].dropna().unique()):
        rgb = name_to_rgb(cand)
        color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:20px;height:20px;background-color:{color_hex};border-radius:3px;'></div>"
            f"{cand.title()}</div>", unsafe_allow_html=True
        )


with tabs[8]:
    st.header("📊 Census Block Group Tables")
    with st.expander("ℹ️ What do the scores mean?"):
        st.markdown("""
    **🧠 Raw Score** – AI-detected narrative alignment in this Census Block Group for the selected candidate  
    **📉 Normalized Score** – Relative strength of support (0–100) within the candidate’s Census Block Group footprint  
    **⚖️ Net Score** – The difference in narrative strength between two candidates in a given Census Block Group 
    **🎯 Priority Score** – Weighted opportunity metric based on:  
    **Normalized Score × Voter Base × Turnout**

    > These scores are based on AI analysis of how each candidate is framed in the news — as a **hero or villain**, with specific **traits, archetypes, issue framing**, and alignment with **socio-demographic and political preferences**.
    """)

    # --- Winner calculation (run once) ---
    pivot_scores = gdf.pivot_table(index=["county_name", "block_group"], columns="candidate", values="score")
    numeric_scores = pivot_scores[candidates]  # only candidate columns
    pivot_scores["precinct_winner"] = numeric_scores.idxmax(axis=1)
    pivot_scores["winner_score"] = numeric_scores.max(axis=1).round(2)
    winner_lookup = pivot_scores[["precinct_winner", "winner_score"]].reset_index()
    gdf = gdf.merge(winner_lookup, on=["county_name", "block_group"], how="left")

    # --- Preprocess for display ---
    gdf["priority"] = (gdf["normalized_score"] * gdf["totalvoters"] * gdf["totalvoterturnout1"]).round(0)
    gdf["score"] = gdf["score"].round(2)
    gdf["normalized_score"] = gdf["normalized_score"].round(2)
    gdf["voter turnout"] = gdf["totalvoterturnout1"].apply(
        lambda x: f"{int(round(x))}%" if pd.notnull(x) else "N/A"
    )

    # --- Individual candidate table ---
    table_cand1 = st.selectbox("Table Candidate A", ["All"] + candidates, index=candidates.index("Jack Ciattarelli") + 1)
    filtered_table = gdf if table_cand1 == "All" else gdf[gdf["candidate"] == table_cand1]

    display_df = filtered_table.rename(columns={"totalvoters": "number of voters"})[[
        "county_name", "block_group", "candidate", "score", "normalized_score",
        "number of voters", "voter turnout", "priority", "precinct_winner", "winner_score"
    ]]
    st.dataframe(display_df.sort_values("priority", ascending=False), use_container_width=True)

    # --- Net score comparison ---
    st.subheader("🆚 Net Score Table")
    cand_x = st.selectbox("Compare A", candidates, index=candidates.index("Jack Ciattarelli"))
    cand_y = st.selectbox("Compare B", candidates, index=candidates.index("Mikie Sherrill"))

    if cand_x != cand_y:
        net_df = gdf[gdf["candidate"].isin([cand_x, cand_y])].pivot_table(
            index=["county_name", "block_group", "totalvoters", "totalvoterturnout1"],
            columns="candidate", values="score"
        ).dropna().reset_index()

        net_df["net_score"] = (net_df[cand_x] - net_df[cand_y]).round(2)
        net_df["number of voters"] = net_df["totalvoters"]
        net_df["voter turnout"] = net_df["totalvoterturnout1"].apply(
            lambda x: f"{int(round(x))}%" if pd.notnull(x) else "N/A"
        )

        # Merge winner info
        net_df = net_df.merge(winner_lookup, on=["county_name", "block_group"], how="left")

        net_display = net_df.rename(columns={
            cand_x: f"{cand_x} Score",
            cand_y: f"{cand_y} Score",
            "winner_score": "Winner Score",
            "precinct_winner": "Precinct Winner"
        })[[
            "county_name", "block_group", f"{cand_x} Score", f"{cand_y} Score",
            "net_score", "Winner Score", "Precinct Winner", "number of voters", "voter turnout"
        ]]
        st.dataframe(net_display.sort_values("net_score", ascending=False), use_container_width=True)

with tabs[9]:
    st.header("🧠 Opposition Research: Census Block Group Alignment & Attack Angles")

    with st.expander("ℹ️ What does the dominant issue map show?"):
        st.markdown("""
        This map shows the **top 1–5 most salient issues** per Census Block Group, based on estimated voter concern levels.
        
        - **Color**: Encodes the selected top issue (e.g., "Inflation and Economy").
        - **Tooltip**: Shows the issue label and salience score (normalized to 100).
        - **Use Case**: Identify which issues dominate attention in different regions.
        """)

    @st.cache_data
    def load_opposition_data():
        with open("data/new_jersey/precinct_issue_alignment_nj.json") as f1, \
             open("data/new_jersey/candidate_position_scores_nj.json") as f2, \
             open("data/new_jersey/opposition_attack_lines_nj.json") as f3, \
             open("data/new_jersey/issue_position_clusters_nj.json") as f4:
            alignment_data = json.load(f1)
            candidate_scores = json.load(f2)
            attack_lines = json.load(f3)
            issue_position_map = json.load(f4)
            normalized_clusters = {}
            for issue, clusters in issue_position_map.items():
                if isinstance(clusters, list):
                    normalized_clusters[issue] = {
                        c["cluster_label"]: c.get("cluster_summary", "") for c in clusters if "cluster_label" in c
                    }
                elif isinstance(clusters, dict):
                    normalized_clusters[issue] = clusters
                else:
                    normalized_clusters[issue] = {}
            issue_position_map = normalized_clusters
        return alignment_data, candidate_scores, attack_lines, issue_position_map

    alignment_data, candidate_scores, attack_lines, issue_position_map = load_opposition_data()
    candidate_list = sorted(candidate_scores.keys())

    st.subheader("📍 Dominant Issue by Census Block Group")
    top_n = st.selectbox("Number of top issues to display per Census Block Group", [1, 2, 3, 4, 5], index=0)

    issue_records = []
    for row in alignment_data:
        salience_dict = row.get("issue_salience", {})
        top_issues = sorted(salience_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for rank, (issue, sal_score) in enumerate(top_issues, 1):
            issue_records.append({
                "county_name": row["county"],
                "block_group": row["precinct"],
                "issue": issue,
                "salience": sal_score,
                "top_issue_rank": f"Top {rank}"
            })

    df_issue = pd.DataFrame(issue_records)
    if not df_issue.empty:
        max_sal = df_issue["salience"].max()
        df_issue["salience"] = (df_issue["salience"] / (max_sal + 1e-6)) * 100
        df_issue["rgb"] = df_issue["issue"].apply(name_to_rgb)
        df_issue[["r", "g", "b", "a"]] = pd.DataFrame(df_issue["rgb"].tolist(), index=df_issue.index)

        map_df = shapes.merge(df_issue, on=["county_name", "block_group"], how="inner")

        def show_pydeck_map(gdf_map, value_col, candidate_name=None, other_score_col=None, second_name=None):

            if gdf_map.empty:
                st.info("No data to show on map.")
                return

            gdf_map = gdf_map.dropna(subset=["geometry"]).copy()
            if other_score_col and other_score_col in gdf_map.columns:
                # Use base hue from issue and brightness from salience
                base_rgb = gdf_map[value_col].apply(name_to_rgb).tolist()
                norm = gdf_map[other_score_col].values / (gdf_map[other_score_col].max() + 1e-6)
                gdf_map["rgb"] = [
                    [int(r * norm[i]), int(g * norm[i]), int(b * norm[i]), 200]
                    for i, (r, g, b, _) in enumerate(base_rgb)
                ]
            else:
                gdf_map["rgb"] = gdf_map[value_col].apply(name_to_rgb)


            gdf_map[["r", "g", "b", "a"]] = pd.DataFrame(gdf_map["rgb"].tolist(), index=gdf_map.index)
            gdf_map = gdf_map.to_crs(epsg=4326)
            geojson = to_geojson_cached(gdf_map, key=f"{value_col}_{candidate_name}_{gdf_map.shape[0]}")

            if "issue" in gdf_map.columns and "salience" in gdf_map.columns:
                tooltip = f"""
                    <b>County:</b> {{county_name}}<br>
                    <b>Precinct:</b> {{block_group}}<br>
                    <b>Top Issue:</b> {{issue}}<br>
                    <b>Salience Score (0–100):</b> {{salience}}<br>
                    <b>Rank:</b> {{top_issue_rank}}
                """
            else:
                tooltip = f"""
                    <b>County:</b> {{county_name}}<br>
                    <b>Precinct:</b> {{block_group}}<br>
                    <b>{value_col.title()}:</b> {{{value_col}}}
                """

            view_state = pdk.ViewState(latitude=39.0, longitude=-105.5, zoom=7.5)
            layer = pdk.Layer("GeoJsonLayer", data=geojson,
                              get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                              get_line_color=[0, 0, 0, 160], pickable=True, auto_highlight=True)

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"html": tooltip, "style": {"backgroundColor": "black", "color": "white"}}
            ))


        show_pydeck_map(map_df, "issue", candidate_name="Top Issue", other_score_col="salience")

    st.subheader("📌 Issue Position Map")
    with st.expander("ℹ️ What does the position map show?"):
        st.markdown("""
        This map displays the **dominant ideological position** on a selected issue in each Census Block Group.
        
        - **Color**: Encodes the dominant stance (e.g., "Progressive", "Moderate").
        - **Tooltip**: Shows the position and a brief summary.
        - **Use Case**: Understand how the public's issue-level ideology varies by Census Block Group.
        """)

    pos_issue = st.selectbox(
        "Select issue for position mapping",
        sorted(issue_position_map.keys()),
        key="pos_map"
    )

    pos_records = []
    for row in alignment_data:
        position_scores = row.get("issue_position_support", {}).get(pos_issue, {})
        cluster_defs = issue_position_map.get(pos_issue)
        if not isinstance(cluster_defs, dict):
            continue

        filtered_scores = {k: v for k, v in position_scores.items() if k in cluster_defs}
        if filtered_scores:
            dominant_position = max(filtered_scores, key=filtered_scores.get)
            summary = cluster_defs.get(dominant_position, "")
            pos_records.append({
                "county_name": row["county"],
                "block_group": row["precinct"],
                "dominant_position": dominant_position,
                "cluster_summary": summary
            })

    pos_df = pd.DataFrame(pos_records)
    if not pos_df.empty:
        pos_df["rgb"] = pos_df["dominant_position"].apply(name_to_rgb)
        pos_df[["r", "g", "b", "a"]] = pd.DataFrame(pos_df["rgb"].tolist(), index=pos_df.index)
        pos_geo = shapes.merge(pos_df, on=["county_name", "block_group"], how="inner")

        def show_position_map(gdf_map):
            gdf_map = gdf_map.dropna(subset=["geometry"]).copy()
            gdf_map = gdf_map.to_crs(epsg=4326)
            geojson = to_geojson_cached(gdf_map, key=f"position_{pos_issue}_{gdf_map.shape[0]}")

            tooltip = """
                <b>County:</b> {county_name}<br>
                <b>Precinct:</b> {block_group}<br>
                <b>Position:</b> {dominant_position}<br>
                <b>Summary:</b> {cluster_summary}
            """

            layer = pdk.Layer("GeoJsonLayer", data=geojson,
                              get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                              get_line_color=[0, 0, 0, 160], pickable=True, auto_highlight=True)

            view_state = pdk.ViewState(latitude=39.0, longitude=-105.5, zoom=7.5)
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    "html": tooltip,
                    "style": {
                        "backgroundColor": "black",
                        "color": "white",
                        "maxWidth": "500px",
                        "whiteSpace": "normal",
                        "wordWrap": "break-word",
                        "overflowWrap": "break-word",
                        "fontSize": "13px"
                    }
                }
            ))

        show_position_map(pos_geo)
    else:
        st.info("No valid position data available for this issue.")
    st.subheader("🆚 A vs B Opportunity/Protect Tables")
    with st.expander("ℹ️ How to interpret Protect and Opportunity Census Block Groups"):
        st.markdown("""
        These tables highlight **high-impact Census Block Groups** based on issue alignment and position contrast between two candidates.

        - **Issue**: One of the top 3 most salient issues in the Census Block Group based on voter concern.
        - **Candidate Position Score**: How closely each candidate's positions align with the **dominant stance** in the Census Block Group.
        - **Advantage**: Gap in alignment — positive means Candidate A has edge.
        - **Protect Score**: Priority score where Candidate A leads but must defend.
        - **Opportunity Score**: Priority score where Candidate A trails but has opening.
        - **Attack Line**: Suggested argument for Candidate A to contrast against B.
        """)

    # ✅ Defaults to Ciattarelli vs. Sherrill
    c1, c2 = st.columns(2)
    cand1 = c1.selectbox("Candidate A (attacker)", candidate_list, index=candidate_list.index("Jack Ciattarelli"), key="ab_attack_c1")
    cand2 = c2.selectbox("Candidate B (target)", candidate_list, index=candidate_list.index("Mikie Sherrill"), key="ab_attack_c2")

    # ✅ Recompute pivot table for scores
    pivot = gdf.pivot_table(
        index=["county_name", "block_group"],
        columns="candidate",
        values="score"
    ).reset_index()

    turnout_map = gdf.groupby(["county_name", "block_group"])[["totalvoters", "totalvoterturnout1"]].first().reset_index()
    pivot = pivot.merge(turnout_map, on=["county_name", "block_group"], how="left")
    pivot["weight"] = pivot["totalvoters"] * pivot["totalvoterturnout1"]
    pivot = pivot.fillna(0)

    def display_enriched_table(mode):
        import ast

        def cluster_similarity(score_dict, cluster):
            return score_dict.get(cluster, 0.0)

        def find_attack_line(attacks, a, b, issue):
            for row in attacks:
                if row["attacker"] == a and row["target"] == b and row["issue"] == issue:
                    try:
                        parsed = ast.literal_eval(row["response"]) if isinstance(row["response"], str) else row["response"]
                        return parsed.get("attack_line", "—")
                    except Exception:
                        return "—"
            return "—"

        records = []
        for row in alignment_data:
            salience = row.get("issue_salience", {})
            positions = row.get("issue_position_support", {})
            top_issues = sorted(salience.items(), key=lambda x: x[1], reverse=True)[:3]  # ✅ Get top 3
            for issue, sal_score in top_issues:
                clusters = positions.get(issue, {})
                if not clusters:
                    continue
                clusters_dict = issue_position_map.get(issue)
                if not isinstance(clusters_dict, dict):
                    continue
                valid_clusters = set(clusters_dict.keys())
                filtered_clusters = {k: v for k, v in clusters.items() if k in valid_clusters}
                if not filtered_clusters:
                    continue
                dom_cluster = max(filtered_clusters, key=filtered_clusters.get)

                score_a = cluster_similarity(candidate_scores.get(cand1, {}).get(issue, {}), dom_cluster)
                score_b = cluster_similarity(candidate_scores.get(cand2, {}).get(issue, {}), dom_cluster)
                advantage = round(score_a - score_b, 2)

                match = pivot[
                    (pivot["county_name"] == row["county"]) & 
                    (pivot["block_group"] == row["precinct"])
                ]
                if not match.empty:
                    s1 = match.iloc[0].get(cand1, 0)
                    s2 = match.iloc[0].get(cand2, 0)
                    w = match.iloc[0]["weight"]
                    margin = max(abs(s1 - s2), 0.01)
                    protect_score = round(w / (1 + margin), 2) if s1 > s2 else 0.0
                    opportunity_score = round((1 / (1 + margin)) * w, 2) if s1 < s2 else 0.0
                else:
                    protect_score = 0.0
                    opportunity_score = 0.0

                records.append({
                    "county": row["county"],
                    "precinct": row["precinct"],
                    "issue": issue,
                    "salience": round((sal_score / (max(salience.values()) + 1e-6)) * 100, 1),
                    f"{cand1}_pos": round(score_a, 2),
                    f"{cand2}_pos": round(score_b, 2),
                    "dominant_position": dom_cluster,
                    "advantage": advantage,
                    "protect_score": protect_score,
                    "opportunity_score": opportunity_score,
                    "attack_line": find_attack_line(attack_lines, cand1, cand2, issue)
                })

        df = pd.DataFrame(records)
        if mode == "protect":
            df = df[df["protect_score"] > 0]
        else:
            df = df[df["opportunity_score"] > 0]

        # ✅ SORT by county, Census Block Group, salience descending
        df = df.sort_values(by=["county", "precinct", "salience"], ascending=[True, True, False])
        st.dataframe(df, use_container_width=True)
        st.download_button(f"Download {mode.title()} Table", df.to_csv(index=False), file_name=f"{cand1}_vs_{cand2}_{mode}.csv")

    st.markdown("### 🛡️ Protect Census Block Group")
    display_enriched_table("protect")

    st.markdown("### 🚀 Opportunity Census Block Group")
    display_enriched_table("opportunity")




# --- Tab 10: Frames ---
with tabs[10]:
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



# --- Tab 11: Topics ---
with tabs[11]:
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