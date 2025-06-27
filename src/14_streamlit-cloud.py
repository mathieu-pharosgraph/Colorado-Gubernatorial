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

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

st.set_page_config(layout="wide")

# --- Logo and Header ---
st.image("src/pharos_logo.png", width=180)
st.markdown("# Model Share: Colorado Narrative Intelligence Dashboard")

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
                continue
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
    "Framing Polarity", "Top Issues", "Framing Wordclouds", "Narrative Insights", "Voter Persuasion Insights", "State Tables", "Precinct Maps", "Precinct Tables", "Opposition Insights", "Frames", "Topics"
])

candidates = sorted(df["candidate"].dropna().unique())

@st.cache_data
def load_precinct_data():
    shapes = gpd.read_file("data/tl_2020_08_vtd20.shp", engine="fiona")

    fips_to_name = {
        "001": "adams", "003": "alamosa", "005": "arapahoe", "007": "archuleta", "009": "baca", "011": "bent",
        "013": "boulder", "014": "broomfield", "015": "chaffee", "017": "cheyenne", "019": "clear creek",
        "021": "conejos", "023": "costilla", "025": "crowley", "027": "custer", "029": "delta", "031": "denver",
        "033": "dolores", "035": "douglas", "037": "eagle", "039": "elbert", "041": "el paso", "043": "fremont",
        "045": "garfield", "047": "gilpin", "049": "grand", "051": "gunnison", "053": "hinsdale", "055": "huerfano",
        "057": "jackson", "059": "jefferson", "061": "kiowa", "063": "kit carson", "065": "lake", "067": "la plata",
        "069": "larimer", "071": "las animas", "073": "lincoln", "075": "logan", "077": "mesa", "079": "mineral",
        "081": "moffat", "083": "montezuma", "085": "montrose", "087": "morgan", "089": "otero", "091": "ouray",
        "093": "park", "095": "phillips", "097": "pitkin", "099": "prowers", "101": "pueblo", "103": "rio blanco",
        "105": "rio grande", "107": "routt", "109": "saguache", "111": "san juan", "113": "san miguel",
        "115": "sedgwick", "117": "summit", "119": "teller", "121": "washington", "123": "weld", "125": "yuma"
    }
    shapes["county_fips"] = shapes["COUNTYFP20"].astype(str).str.zfill(3)
    shapes["county_name"] = shapes["county_fips"].map(fips_to_name)
    shapes["precinct_num"] = shapes["VTDST20"].astype(str).str.zfill(6)
    shapes["precinct_code"] = shapes["precinct_num"].str[-3:]
    shapes["geometry"] = shapes["geometry"].simplify(0.001, preserve_topology=True)

    shapes = shapes[["county_name", "precinct_code", "geometry"]]

    scores = pd.read_csv("data/precinct_candidate_scores_with_confidence.csv")
    scores["precinct_num"] = scores["precinct_num"].astype(str).str.zfill(3)
    scores["county_name"] = scores["county_name"].str.strip().str.lower()
    scores["precinct_code"] = scores["precinct_num"]

    turnout = pd.read_stata("data/socio_demo_politics_precinct_final.dta")
    turnout["precinct_num"] = turnout["precinct_num"].astype(str).str.zfill(3)
    turnout["county_name"] = turnout["county_name"].str.strip().str.lower()
    turnout["precinct_code"] = turnout["precinct_num"]

    merged = shapes.merge(scores, on=["county_name", "precinct_code"], how="inner")
    merged = merged.merge(turnout[["county_name", "precinct_code", "totalvoters", "totalvoterturnout1"]],
                          on=["county_name", "precinct_code"], how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=shapes.crs), shapes



gdf, shapes = load_precinct_data()



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
        <b>Precinct:</b> {{precinct_code}}<br>
        <b>{value_col.title()}:</b> {{{value_col}}}
    """
    if value_col == "net_score" and second_name:
        tooltip = f"""
            <b>{candidate_name}:</b> {{cand1_score}}<br>
            <b>{second_name}:</b> {{cand2_score}}<br>
            <b>Net Score:</b> {{net_score}}<br>
            <b>County:</b> {{county_name}}<br>
            <b>Precinct:</b> {{precinct_code}}
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
        **📊 State Score** – Weighted average narrative strength of each candidate, where each precinct contributes:  
        **Raw Score × Number of Voters × Turnout**  
        normalized by total **Voters × Turnout** across precincts with scores.

        This reflects **narrative alignment scaled by potential electoral impact**.
        """)

    # Calculate state-level weighted scores
    gdf_valid = gdf[gdf["score"].notnull()]
    gdf_valid["weight"] = gdf_valid["totalvoters"] * gdf_valid["totalvoterturnout1"]

    state_scores = gdf_valid.groupby("candidate").apply(
        lambda g: (g["score"] * g["weight"]).sum() / g["weight"].sum()
    ).round(2).reset_index(name="State Score")

    st.subheader("🏅 State Score Rankings")
    st.dataframe(state_scores.sort_values("State Score", ascending=False), use_container_width=True)

    # Head-to-head filters
    st.subheader("🤜 Head-to-Head Analysis")
    col1, col2 = st.columns(2)
    cand1 = col1.selectbox("Candidate A", candidates, index=candidates.index("Michael Bennet"))
    cand2 = col2.selectbox("Candidate B", candidates, index=candidates.index("Phil Weiser"))

    if cand1 != cand2:
        pivot = gdf[gdf["candidate"].isin([cand1, cand2])].pivot_table(
            index=["county_name", "precinct_code", "totalvoters", "totalvoterturnout1"],
            columns="candidate", values="score"
        ).dropna().reset_index()

        # Compute protection and opportunity scores
        pivot["weight"] = pivot["totalvoters"] * pivot["totalvoterturnout1"]
        margin = (pivot[cand1] - pivot[cand2]).clip(lower=0.01)  # avoid divide-by-zero
        pivot["protect_score"] = ((pivot[cand1] > pivot[cand2]).astype(int)) * (pivot["weight"] / (1 + margin))
        pivot["opportunity_score"] = ((pivot[cand1] < pivot[cand2]).astype(int)) * (1 / (1 + (pivot[cand2] - pivot[cand1]).abs())) * pivot["weight"]

        # Protect table
        st.subheader(f"🛡️ Precincts to Protect for {cand1}")
        with st.expander("ℹ️ What do the precincts to protect scores mean?"):
            st.markdown("""
            **📊 Protect Score** – Considers only the precinct where the first candidate LEADS the second candidates. Score then based on number of voters * turnout weighted by the inverse of how much the first candidate leads in that precinct (i.e., precinct where the first candidate leads by only a small margin will show higher scores (for the same voters * turnout)). 

            """)
        protect_table = pivot[pivot["protect_score"] > 0][[
            "county_name", "precinct_code", cand1, cand2, "protect_score"
        ]].rename(columns={
            cand1: f"{cand1} Score",
            cand2: f"{cand2} Score"
        }).sort_values("protect_score", ascending=False)
        st.dataframe(protect_table, use_container_width=True)

        # Opportunity table
        st.subheader(f"🚀 Opportunity Precincts for {cand1}")
        with st.expander("ℹ️ What do the opportinity precincts scores mean?"):
            st.markdown("""
            **📊 Opportunity Score** – Considers only the precinct where the first candidate LAGS the second candidates. Score then based on number of voters * turnout weighted by the inverse of how much the first candidate lags in that precinct (i.e., precinct where the first candidate lags by only a small margin will show higher scores (for the same voters * turnout)). 

            """)
        opportunity_table = pivot[pivot["opportunity_score"] > 0][[
            "county_name", "precinct_code", cand1, cand2, "opportunity_score"
        ]].rename(columns={
            cand1: f"{cand1} Score",
            cand2: f"{cand2} Score"
        }).sort_values("opportunity_score", ascending=False)
        st.dataframe(opportunity_table, use_container_width=True)


with tabs[7]:
    with st.expander("ℹ️ What do the scores mean?"):
        st.markdown("""
    **🧠 Raw Score** – AI-detected narrative alignment in this precinct for the selected candidate  
    **📉 Normalized Score** – Relative strength of support (0–100) within the candidate’s precinct footprint  
    **⚖️ Net Score** – The difference in narrative strength between two candidates in a given precinct  
    **🎯 Priority Score** – Weighted opportunity metric based on:  
    **Normalized Score × Voter Base × Turnout**

    > These scores are based on AI analysis of how each candidate is framed in the news — as a **hero or villain**, with specific **traits, archetypes, issue framing**, and alignment with **socio-demographic and political preferences**.
    """)
    st.header("🗺️ Precinct Score Maps")
    candidates = sorted(gdf["candidate"].dropna().unique())
    cand1 = st.selectbox("Map Candidate A", ["All"] + candidates, index=candidates.index("Michael Bennet") + 1)
    cand2 = st.selectbox("Map Candidate B", ["All"] + candidates, index=candidates.index("Phil Weiser") + 1)
    filtered = gdf.copy()

    # Pivot table with all candidate scores for winner map later
    pivot_all = gdf.pivot(index=["county_name", "precinct_code"], columns="candidate", values="score").reset_index()

    # --- Map for Candidate A ---
    if cand1 != "All":
        st.subheader(f"🗺️ Precinct Scores for {cand1}")
        pivot1 = gdf[gdf["candidate"] == cand1][["county_name", "precinct_code", "score"]].copy()
        pivot1 = pivot1.rename(columns={"score": "cand1_score"})
        df1 = shapes.copy().merge(pivot1, on=["county_name", "precinct_code"], how="inner")
        df1 = df1.rename(columns={"cand1_score": "score"})
        show_pydeck_map(df1, "score", candidate_name=cand1)

    # --- Map for Candidate B ---
    if cand2 != "All":
        st.subheader(f"🗺️ Precinct Scores for {cand2}")
        pivot2 = gdf[gdf["candidate"] == cand2][["county_name", "precinct_code", "score"]].copy()
        pivot2 = pivot2.rename(columns={"score": "cand2_score"})
        df2 = shapes.copy().merge(pivot2, on=["county_name", "precinct_code"], how="inner")
        df2 = df2.rename(columns={"cand2_score": "score"})
        show_pydeck_map(df2, "score", candidate_name=cand2)





    if cand1 != "All" and cand2 != "All" and cand1 != cand2:
        wide = filtered.pivot(index=["county_name", "precinct_code"], columns="candidate", values="score").reset_index()
        wide = wide.dropna(subset=[cand1, cand2])
        wide["net_score"] = (wide[cand1] - wide[cand2]).round(2)
        wide["cand1_score"] = wide[cand1].round(2)
        wide["cand2_score"] = wide[cand2].round(2)
        geo_net = shapes[["county_name", "precinct_code", "geometry"]].merge(
            wide[["county_name", "precinct_code", "net_score", "cand1_score", "cand2_score"]],
            on=["county_name", "precinct_code"],
            how="inner"
        )

        st.subheader(f"Net Support: {cand1} – {cand2}")
        show_pydeck_map(geo_net, "net_score", candidate_name=cand1, second_name=cand2)

    # --- 🏆 Winner Map ---
    st.subheader("🏆 Winner Map (Top Score)")


    winner_df = gdf.copy()
    pivot = winner_df.pivot_table(index=["county_name", "precinct_code"], columns="candidate", values="score")
    score_only = pivot[candidates]  # restrict to candidate score columns
    pivot["winner"] = score_only.idxmax(axis=1)
    pivot["max_score"] = score_only.max(axis=1).round(2)


    winner_geo = shapes.merge(pivot.reset_index(), on=["county_name", "precinct_code"], how="inner")
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
    st.header("📊 Precinct Tables")
    with st.expander("ℹ️ What do the scores mean?"):
        st.markdown("""
    **🧠 Raw Score** – AI-detected narrative alignment in this precinct for the selected candidate  
    **📉 Normalized Score** – Relative strength of support (0–100) within the candidate’s precinct footprint  
    **⚖️ Net Score** – The difference in narrative strength between two candidates in a given precinct  
    **🎯 Priority Score** – Weighted opportunity metric based on:  
    **Normalized Score × Voter Base × Turnout**

    > These scores are based on AI analysis of how each candidate is framed in the news — as a **hero or villain**, with specific **traits, archetypes, issue framing**, and alignment with **socio-demographic and political preferences**.
    """)

    # --- Winner calculation (run once) ---
    pivot_scores = gdf.pivot_table(index=["county_name", "precinct_code"], columns="candidate", values="score")
    numeric_scores = pivot_scores[candidates]  # only candidate columns
    pivot_scores["precinct_winner"] = numeric_scores.idxmax(axis=1)
    pivot_scores["winner_score"] = numeric_scores.max(axis=1).round(2)
    winner_lookup = pivot_scores[["precinct_winner", "winner_score"]].reset_index()
    gdf = gdf.merge(winner_lookup, on=["county_name", "precinct_code"], how="left")

    # --- Preprocess for display ---
    gdf["priority"] = (gdf["normalized_score"] * gdf["totalvoters"] * gdf["totalvoterturnout1"]).round(0)
    gdf["score"] = gdf["score"].round(2)
    gdf["normalized_score"] = gdf["normalized_score"].round(2)
    gdf["voter turnout"] = gdf["totalvoterturnout1"].apply(
        lambda x: f"{int(round(x))}%" if pd.notnull(x) else "N/A"
    )

    # --- Individual candidate table ---
    table_cand1 = st.selectbox("Table Candidate A", ["All"] + candidates, index=candidates.index("Michael Bennet") + 1)
    filtered_table = gdf if table_cand1 == "All" else gdf[gdf["candidate"] == table_cand1]

    display_df = filtered_table.rename(columns={"totalvoters": "number of voters"})[[
        "county_name", "precinct_code", "candidate", "score", "normalized_score",
        "number of voters", "voter turnout", "priority", "precinct_winner", "winner_score"
    ]]
    st.dataframe(display_df.sort_values("priority", ascending=False), use_container_width=True)

    # --- Net score comparison ---
    st.subheader("🆚 Net Score Table")
    cand_x = st.selectbox("Compare A", candidates, index=candidates.index("Michael Bennet"))
    cand_y = st.selectbox("Compare B", candidates, index=candidates.index("Phil Weiser"))

    if cand_x != cand_y:
        net_df = gdf[gdf["candidate"].isin([cand_x, cand_y])].pivot_table(
            index=["county_name", "precinct_code", "totalvoters", "totalvoterturnout1"],
            columns="candidate", values="score"
        ).dropna().reset_index()

        net_df["net_score"] = (net_df[cand_x] - net_df[cand_y]).round(2)
        net_df["number of voters"] = net_df["totalvoters"]
        net_df["voter turnout"] = net_df["totalvoterturnout1"].apply(
            lambda x: f"{int(round(x))}%" if pd.notnull(x) else "N/A"
        )

        # Merge winner info
        net_df = net_df.merge(winner_lookup, on=["county_name", "precinct_code"], how="left")

        net_display = net_df.rename(columns={
            cand_x: f"{cand_x} Score",
            cand_y: f"{cand_y} Score",
            "winner_score": "Winner Score",
            "precinct_winner": "Precinct Winner"
        })[[
            "county_name", "precinct_code", f"{cand_x} Score", f"{cand_y} Score",
            "net_score", "Winner Score", "Precinct Winner", "number of voters", "voter turnout"
        ]]
        st.dataframe(net_display.sort_values("net_score", ascending=False), use_container_width=True)

with tabs[9]:
    st.header("🧠 Opposition Research: Precinct Alignment & Attack Angles")

    # Load data (once at top of file or here with cache)
    @st.cache_data
    def load_opposition_data():
        with open("data/precinct_issue_alignment.json") as f1, \
            open("data/candidate_position_scores.json") as f2, \
            open("data/opposition_attack_lines.json") as f3, \
            open("data/issue_position_clusters.json") as f4:
            alignment_data = json.load(f1)
            candidate_scores = json.load(f2)
            attack_lines = json.load(f3)
            issue_position_map = json.load(f4)
            # Normalize structure to: issue → cluster_label → summary
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

    # Replace this entire block starting from `st.subheader("\ud83d\udccd Dominant Issue by Precinct")`

    st.subheader("📍 Dominant Issue by Precinct")
    top_n = st.selectbox("Number of top issues to display per precinct", [1, 2, 3, 4, 5], index=0)
    pos_issue = st.selectbox("Select issue for position mapping", sorted(alignment_data[0]["issue_position_support"].keys()), key="pos_map")
    # Build top-N issue records
    issue_records = []
    pos_records = []
    for row in alignment_data:
        position_scores = row.get("issue_position_support", {}).get(pos_issue, {})
        
        # Get valid cluster summaries
        clusters_dict = issue_position_map.get(pos_issue, {})
        filtered_scores = {k: v for k, v in position_scores.items() if k in clusters_dict}

        if filtered_scores:
            dominant_position = max(filtered_scores, key=filtered_scores.get)
            summary = clusters_dict.get(dominant_position, "No summary available")
            pos_records.append({
                "county_name": row["county"],
                "precinct_code": row["precinct"],
                "dominant_position": dominant_position,
                "cluster_summary": summary
            })


    df_dominant = pd.DataFrame(issue_records)
    if not df_dominant.empty:
        max_sal = df_dominant["salience"].max()
        df_dominant["salience"] = (df_dominant["salience"] / (max_sal + 1e-6)) * 100
    df_dominant["rgb"] = df_dominant["dominant_position"].apply(name_to_rgb)
    df_dominant[["r", "g", "b", "a"]] = pd.DataFrame(df_dominant["rgb"].tolist(), index=df_dominant.index)

    # Merge and show
    map_df = shapes.merge(df_dominant, on=["county_name", "precinct_code"], how="inner")
    show_pydeck_map(map_df, "issue", candidate_name="Top Issue")

    # Fix tooltip in show_pydeck_map()
    def show_pydeck_map(gdf_map, value_col, candidate_name=None, other_score_col=None, second_name=None):
        if gdf_map.empty:
            st.info("No data to show on map.")
            return

        gdf_map = gdf_map.dropna(subset=["geometry"]).copy()

        if pd.api.types.is_numeric_dtype(gdf_map[value_col]):
            gdf_map[value_col] = gdf_map[value_col].round(2)
            gdf_map["score_norm"] = gdf_map[value_col] / (gdf_map[value_col].max() + 1e-6)
            gdf_map["rgb"] = gdf_map["score_norm"].apply(lambda x: [int(255 * (1 - x)), int(255 * x), 50, 200])
        else:
            gdf_map["rgb"] = gdf_map[value_col].apply(name_to_rgb)

        gdf_map[["r", "g", "b", "a"]] = pd.DataFrame(gdf_map["rgb"].tolist(), index=gdf_map.index)
        gdf_map = gdf_map.to_crs(epsg=4326)
        geojson = to_geojson_cached(gdf_map, key=f"{value_col}_{candidate_name}_{gdf_map.shape[0]}")

        if "issue" in gdf_map.columns and "salience" in gdf_map.columns:
            tooltip = f"""
                <b>County:</b> {{county_name}}<br>
                <b>Precinct:</b> {{precinct_code}}<br>
                <b>Top Issue:</b> {{issue}}<br>
                <b>Salience:</b> {{salience:.1f}}
            """
        elif value_col == "dominant_position":
            tooltip = f"""
                <b>County:</b> {{county_name}}<br>
                <b>Precinct:</b> {{precinct_code}}<br>
                <b>Position:</b> {{{value_col}}}<br>
                <b>Summary:</b> {{cluster_summary}}
            """

        else:
            tooltip = f"""
                <b>County:</b> {{county_name}}<br>
                <b>Precinct:</b> {{precinct_code}}<br>
                <b>{value_col.title()}:</b> {{{value_col}}}
            """


        view_state = pdk.ViewState(latitude=39.0, longitude=-105.5, zoom=7.5)
        layer = pdk.Layer("GeoJsonLayer", data=geojson,
                        get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                        get_line_color=[0, 0, 0, 160], pickable=True, auto_highlight=True)

        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state,
                                layers=[layer], tooltip={"html": tooltip, "style": {"backgroundColor": "black", "color": "white"}}))


    st.subheader("📌 Issue Position Map")
    
    pos_records = []
    for row in alignment_data:
        position_scores = row.get("issue_position_support", {}).get(pos_issue, {})
        
        # Validate issue-specific cluster space
        issue_clusters_dict = issue_position_map.get(pos_issue)
        if not isinstance(issue_clusters_dict, dict):
            st.warning(f"⚠️ Cluster definitions missing or malformed for: {pos_issue}")
            continue  # or skip rendering
        valid_clusters = set(issue_clusters_dict.keys())

        filtered_scores = {k: v for k, v in position_scores.items() if k in valid_clusters}

        if filtered_scores:
            dominant_position = max(filtered_scores, key=filtered_scores.get)
            issue_records.append({
                "county_name": row["county"],
                "precinct_code": row["precinct"],
                "dominant_position": dominant_position,
                "cluster_summary": summary
            })

        else:
            continue  # Skip if no valid clusters

    pos_df = pd.DataFrame(pos_records)

    # Add color and merge with shapes
    pos_df["rgb"] = pos_df["dominant_position"].apply(name_to_rgb)
    pos_df[["r", "g", "b", "a"]] = pd.DataFrame(pos_df["rgb"].tolist(), index=pos_df.index)
    pos_geo = shapes.merge(pos_df, on=["county_name", "precinct_code"], how="inner")
    show_pydeck_map(pos_geo, "dominant_position")


    st.subheader("🆚 A vs B Opportunity/Protect Tables")
    c1, c2 = st.columns(2)
    cand1 = c1.selectbox("Candidate A (attacker)", candidate_list, index=0, key="ab_attack_c1")
    cand2 = c2.selectbox("Candidate B (target)", candidate_list, index=1, key="ab_attack_c2")

    def display_enriched_table(mode):
        from difflib import SequenceMatcher

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
            salience = row["issue_salience"]
            positions = row["issue_position_support"]
            top_issues = sorted(salience.items(), key=lambda x: x[1], reverse=True)[:3]

            for issue, sal_score in top_issues:
                clusters = positions.get(issue, {})
                if not clusters:
                    continue

                # ✅ Filter to valid cluster labels for this issue
                clusters_dict = issue_position_map.get(issue)
                if not isinstance(clusters_dict, dict):
                    continue  # skip invalid or missing issue
                valid_clusters = set(clusters_dict.keys())

                filtered_clusters = {k: v for k, v in clusters.items() if k in valid_clusters}
                if not filtered_clusters:
                    continue

                dom_cluster = max(filtered_clusters, key=filtered_clusters.get)

                score_a = cluster_similarity(candidate_scores.get(cand1, {}).get(issue, {}), dom_cluster)
                score_b = cluster_similarity(candidate_scores.get(cand2, {}).get(issue, {}), dom_cluster)

                advantage = round(score_a - score_b, 2)
                if (mode == "protect" and advantage > 0) or (mode == "opportunity" and advantage < 0):
                    records.append({
                        "county": row["county"],
                        "precinct": row["precinct"],
                        "issue": issue,
                        "salience": round(sal_score, 2),
                        f"{cand1}_pos": round(score_a, 2),
                        f"{cand2}_pos": round(score_b, 2),
                        "dominant_position": dom_cluster,
                        "advantage": advantage,
                        "attack_line": find_attack_line(attack_lines, cand1, cand2, issue)
                    })

        df = pd.DataFrame(records)
        st.dataframe(df.sort_values("advantage", ascending=(mode == "opportunity")), use_container_width=True)
        st.download_button(f"Download {mode} table", df.to_csv(index=False), file_name=f"{cand1}_vs_{cand2}_{mode}_table.csv")

    st.markdown("### 🛡️ Protect Precincts")
    display_enriched_table("protect")

    st.markdown("### 🚀 Opportunity Precincts")
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