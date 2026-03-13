import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Talent Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Talent Intelligence Agent")
st.caption("AI-powered job market insights · 123K LinkedIn postings · Built by Aditi Khare")

# ── Load Data & Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("jobs_clean.csv")
    return df

@st.cache_resource
def build_index(_df, _model):
    texts = _df["embed_text"].fillna("").tolist()
    embeddings = _model.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

model = load_model()
df = load_data()
index, embeddings = build_index(df, model)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Job Search", "📊 Skills Gap Analyzer", "🤖 Ask the Agent"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SEMANTIC JOB SEARCH
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Semantic Job Search")
    st.write("Describe your skills and find matching roles from analyst job postings.")

    query = st.text_input("Enter your skills or job description",
                          placeholder="e.g. Python SQL Tableau data analyst dashboard")
    top_k = st.slider("Number of results", 3, 20, 10)

    if st.button("Search Jobs", key="search"):
        if query.strip():
            with st.spinner("Searching..."):
                query_vec = model.encode([query]).astype("float32")
                faiss.normalize_L2(query_vec)
                distances, indices = index.search(query_vec, top_k)
                results = df.iloc[indices[0]].copy()
                results["similarity_score"] = np.round(distances[0], 2)

            st.success(f"Top {top_k} matches found!")
            for _, row in results.iterrows():
                score = row['similarity_score']
                if score >= 0.85:
                    quality = "🟢 Excellent"
                elif score >= 0.75:
                    quality = "🟡 Good"
                else:
                    quality = "🟠 Fair"
                with st.expander(f"💼 {row.get('title','N/A')} @ {row.get('company_name','N/A')} — {quality} ({score})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📍 **Location:** {row.get('location','N/A')}")
                        st.write(f"🎯 **Experience:** {row.get('formatted_experience_level','N/A')}")
                    with col2:
                        st.write(f"⭐ **Match Score:** {score}")
                    desc = str(row.get('description',''))[:400]
                    st.write(f"📝 {desc}...")
        else:
            st.warning("Please enter a search query!")

# ══════════════════════════════════════════════════════════════
# TAB 2 — SKILLS GAP ANALYZER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Skills Gap Analyzer")
    st.write("Compare your skills against what the market demands for your target role.")

    col1, col2 = st.columns(2)
    with col1:
        target_role = st.selectbox("Target Role",
            ["Data Analyst", "Business Analyst", "People Analyst",
             "Financial Analyst", "HR Analyst"])
    with col2:
        user_skills_input = st.text_input("Your Skills (comma-separated)",
            placeholder="e.g. Python, SQL, Tableau, Excel, Power BI")

    if st.button("Analyze Gap", key="gap"):
        if user_skills_input.strip():
            user_skills = [s.strip().lower() for s in user_skills_input.split(",")]
            role_jobs = df[df["title"].str.contains(target_role, case=False, na=False)]

            skill_keywords = ["sql","python","tableau","power bi","excel","machine learning",
                              "statistics","workday","looker","databricks","snowflake",
                              "aws","azure","jira","r","sap","dbt","airflow","salesforce"]

            desc = role_jobs["description"].fillna("").str.lower()
            market_skills = {s: desc.str.contains(rf"\b{s}\b", na=False).sum()
                             for s in skill_keywords}
            top_market = set([s for s, _ in sorted(market_skills.items(),
                                                     key=lambda x: x[1], reverse=True)[:10]])
            user_set = set(user_skills)
            have = user_set & top_market
            gaps = top_market - user_set

            st.subheader(f"📋 Results for {target_role} ({len(role_jobs)} postings)")

            col1, col2, col3 = st.columns(3)
            col1.metric("Match Score", f"{len(have)}/{len(top_market)}")
            col2.metric("Skills You Have ✅", len(have))
            col3.metric("Skills to Learn 📚", len(gaps))

            col_a, col_b = st.columns(2)
            with col_a:
                st.success("✅ Skills You Have")
                for s in sorted(have):
                    st.write(f"• {s.upper()}")
            with col_b:
                st.error("📚 Skills to Learn")
                for s in sorted(gaps):
                    st.write(f"• {s.upper()}")

            skills_df = pd.DataFrame(
                sorted(market_skills.items(), key=lambda x: x[1], reverse=True)[:10],
                columns=["Skill", "Demand"]
            )
            st.bar_chart(skills_df.set_index("Skill"))
        else:
            st.warning("Please enter your skills!")

# ══════════════════════════════════════════════════════════════
# TAB 3 — SMART AGENT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Ask the Talent Agent")
    st.write("Ask natural language questions about the job market.")

    example_questions = [
        "What skills do I need for a People Analyst role?",
        "What is the salary range for a data analyst?",
        "What is the minimum salary of an analyst in Arizona?",
        "Which locations have the most business analyst jobs?",
        "What companies are hiring business analysts remotely?"
    ]
    selected = st.selectbox("Try an example:", ["-- Type your own --"] + example_questions)
    question = st.text_input("Your question:",
                              value="" if selected == "-- Type your own --" else selected)

    if st.button("Ask Agent", key="agent"):
        if question.strip():
            with st.spinner("Thinking..."):
                q = question.lower()

                # Location detection
                location_keywords = {
                    "arizona": "AZ", "phoenix": "Phoenix",
                    "new york": "NY", "new jersey": "NJ",
                    "california": "CA", "los angeles": "Los Angeles",
                    "san francisco": "San Francisco",
                    "texas": "TX", "dallas": "Dallas", "houston": "Houston",
                    "florida": "FL", "miami": "Miami",
                    "chicago": "Chicago", "seattle": "Seattle",
                    "boston": "Boston", "denver": "Denver",
                    "atlanta": "Atlanta", "charlotte": "Charlotte",
                    "washington": "Washington", "virginia": "VA"
                }
                detected_location = None
                loc_filter = None
                for keyword, loc in location_keywords.items():
                    if keyword in q:
                        detected_location = keyword.title()
                        loc_filter = loc
                        break

                # FAISS cosine search
                query_vec = model.encode([question]).astype("float32")
                faiss.normalize_L2(query_vec)
                distances, indices = index.search(query_vec, 5)
                results = df.iloc[indices[0]].copy()

                if "salary" in q:
                    salary_data = df[df["title"].str.contains(
                        "analyst", case=False, na=False)].copy()
                    salary_data = salary_data.dropna(subset=["min_salary","max_salary"])
                    salary_data = salary_data[salary_data["max_salary"] < 500000]

                    if loc_filter:
                        salary_data = salary_data[salary_data["location"].str.contains(
                            loc_filter, case=False, na=False)]

                    if len(salary_data) > 0:
                        low = int(salary_data["min_salary"].median())
                        high = int(salary_data["max_salary"].median())
                        loc_str = f" in {detected_location}" if detected_location else ""
                        answer = (f"Based on {len(salary_data)} analyst postings{loc_str} "
                                  f"with salary data, the typical range is "
                                  f"${low//1000}K-${high//1000}K/year. "
                                  f"Mid-Senior level roles average significantly higher than entry level.")
                        if loc_filter:
                            results = salary_data.head(5)
                    else:
                        answer = (f"Not enough salary data found for {detected_location or 'that location'}. "
                                  f"Try a broader search like 'salary range for data analyst'.")

                elif "skill" in q or "require" in q or "need" in q:
                    role = next((r for r in ["people analyst","data analyst",
                                             "business analyst","financial analyst",
                                             "hr analyst"] if r in q), "analyst")
                    role_jobs = df[df["title"].str.contains(role, case=False, na=False)]
                    skill_keywords = ["sql","python","tableau","power bi","excel",
                                      "machine learning","statistics","workday",
                                      "looker","databricks","snowflake","aws","azure"]
                    desc = role_jobs["description"].fillna("").str.lower()
                    market_skills = {s: desc.str.contains(rf"\b{s}\b", na=False).sum()
                                     for s in skill_keywords}
                    top5 = [s.upper() for s, _ in sorted(market_skills.items(),
                                                          key=lambda x: x[1], reverse=True)[:5]]
                    answer = (f"For {role.title()} roles ({len(role_jobs)} postings), "
                              f"the top 5 most demanded skills are: {', '.join(top5)}.")

                elif "location" in q or "where" in q or "city" in q:
                    top_locs = df[df["title"].str.contains(
                        "analyst", case=False, na=False)]["location"]\
                        .dropna().value_counts().head(5).index.tolist()
                    answer = f"Top locations for analyst jobs: {', '.join(top_locs)}."

                elif "compan" in q or "hiring" in q or "employer" in q:
                    top_cos = df[df["title"].str.contains(
                        "analyst", case=False, na=False)]["company_name"]\
                        .dropna().value_counts().head(5).index.tolist()
                    answer = f"Top companies hiring analysts: {', '.join(top_cos)}."

                elif "remote" in q:
                    remote_jobs = df[df["location"].str.contains(
                        "remote|united states", case=False, na=False)]
                    top_cos = remote_jobs["company_name"].dropna()\
                        .value_counts().head(5).index.tolist()
                    answer = (f"Found {len(remote_jobs)} remote/US-wide postings. "
                              f"Top hiring companies: {', '.join(top_cos)}.")

                else:
                    top_title = results.iloc[0]["title"]
                    top_company = results.iloc[0]["company_name"]
                    top_loc = results.iloc[0]["location"]
                    answer = (f"Found {len(results)} relevant roles. "
                              f"Top match: {top_title} at {top_company} in {top_loc}. "
                              f"Try asking about skills, salary, location, or companies!")

            st.subheader("💡 Answer")
            st.success(answer)

            st.subheader("📄 Source Jobs Used")
            st.dataframe(results[["title","company_name","location",
                                   "formatted_experience_level"]].reset_index(drop=True))
        else:
            st.warning("Please ask a question!")
