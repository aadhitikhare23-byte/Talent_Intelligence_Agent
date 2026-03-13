# Talent Intelligence Agent

Ever wished you could just ask the job market 
a question and get a real answer?

There are tools out there — but I wanted to 
build one myself, using real data and learn 
what actually goes into it.

This AI agent lets you search 123,000+ real 
LinkedIn job postings in plain English — type 
what you're looking for, pick your mode, 
and actually get an answer.



**Built by:** Aditi Khare | MS Business Analytics, Arizona State University  
**Dataset:** [LinkedIn Job Postings on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

---

## What Can You Ask It?

Things like:
- *"What skills do I need for a People Analyst role?"*
- *"What's the salary range for a Data Analyst?"*
- *"Which companies are hiring Business Analysts remotely?"*
- *"Where are most analyst jobs located?"*
- *"What experience level do most Data Analyst roles require?"*

---

## How It Works

It's not just keyword search. The agent understands the *meaning* of your question,
finds the most relevant job postings, and gives you a direct, data-backed answer.

```
You ask a question
        ↓
Agent encodes it into a vector (semantic understanding)
        ↓
FAISS searches 1,653 analyst job postings for closest matches
        ↓
Smart answer engine extracts the insight you actually need
        ↓
You get a clean, specific answer
```

No API key needed. Everything runs locally for free.


---

## What I Used

| What | Why |
|---|---|
| FAISS | Lightning-fast semantic search across job postings |
| all-MiniLM-L6-v2 | Turns job descriptions into searchable vectors |
| google/flan-t5-base | Local LLM — free, no quota limits |
| LangGraph | Connects the retrieval and generation steps as an agent |
| MLflow | Tracks experiment runs and similarity scores |
| Plotly | Clean, interactive EDA charts |
| Google Colab T4 GPU | Free GPU for fast embedding |

---

## What the Data Showed

I started with 123,849 raw job postings and filtered down to 1,653 
analyst-specific roles. Here's what I found:

- **Most demanded skill:** Excel (yes, really — still beats Python in job postings)
- **Data Analyst salary range:** $85,000 – $110,125 (median ~$95K/year)
- **Top hiring cities:** New York, Chicago, Atlanta, Houston
- **Most active companies:** Insight Global, Talentify.io, ATC, Lightcast
- **Most common level:** Mid-Senior roles dominate the market

> *Note: 75% of postings don't list salary. I intentionally didn't fill those 
> gaps with averages — that would be misleading. Salary stats only reflect 
> postings that actually disclosed compensation.*

---

## Skills Gap Analyzer

One of my favorite features — you give it your current skills and a target role,
and it tells you what you have vs. what the market actually wants:

```python
skills_gap_analyzer(
    user_skills=['sql', 'python', 'tableau', 'power bi', 'excel'],
    target_role='Data Analyst'
)

# ✅ Skills you HAVE:     SQL, PYTHON, TABLEAU, EXCEL, POWER BI
# ❌ Skills you're MISSING: STATISTICS, SNOWFLAKE, DATABRICKS
# 📊 Match Score: 5/8

``` 

## How Well Does It Search?
I tracked similarity scores across test queries using MLflow:

| Question                  | Avg Similarity Score |
| ------------------------- | -------------------- |
| People Analyst skills     | 0.54                 |
| Remote business analysts  | 0.60                 |
| Experience level required | 0.58                 |
| Location distribution     | 0.54                 |

Scores between 0.5–0.6 are solid for this type of semantic search —
it means the agent is consistently finding relevant postings, not random ones.

## Run It Yourself

**On Google Colab (easiest)**
1. Open `Talent_Intelligence_Agent.ipynb` in Colab
2. Set runtime to **T4 GPU** → Runtime → Change runtime type
3. Add your Kaggle credentials to Colab Secrets:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
4. Run all cells — takes about 5 minutes end to end

**Locally**
git clone https://github.com/YOUR_USERNAME/talent-intelligence-agent
cd talent-intelligence-agent
pip install -r requirements.txt
jupyter notebook Talent_Intelligence_Agent.ipynb


## What's in This Repo

```
talent-intelligence-agent/
├── Talent_Intelligence_Agent.ipynb  ← the whole project lives here
├── jobs_clean.csv                    ← 1,653 cleaned analyst job postings
├── requirements.txt                  ← everything you need to install
└── README.md                         ← you're reading it
```


## 🔗 Live App

Try it here — no code, no setup needed:
[talentintelligenceagent-ffryfnazrn9e4idnbv7udc.streamlit.app](https://talentintelligenceagent-ffryfnazrn9e4idnbv7udc.streamlit.app)



**Let's Connect**

Aditi Khare| [LinkedIn] https://www.linkedin.com/in/aadhitikhare23/ | [GitHub] https://github.com/aadhitikhare23-byte

Open to Data Analyst, People Analyst, Business Intelligence and Business Analyst roles — feel free to reach out!
