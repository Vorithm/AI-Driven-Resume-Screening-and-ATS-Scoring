import streamlit as st
from PyPDF2 import PdfReader
import re
import spacy
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from io import BytesIO
from fpdf import FPDF
import datetime
from dateutil import parser
from rank_bm25 import BM25Okapi  
import datefinder  
import torch
from sentence_transformers import SentenceTransformer
import os 


# Setup NLTK stopwords
nltk.download('stopwords', quiet=True)
EN_STOPWORDS = set(nltk_stopwords.words('english'))

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # fallback if somehow not found
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Skill dictionary (extended)
new_top_skills = [
    "algorithms", "analytical", "analytical skills", "analytics", "artificial intelligence", "aws",
    "azure", "beautiful soup", "big data", "business intelligence", "c++", "cloud", 
    "communication", "computer science", "computer vision", "css", "data analysis", "data analyst",
    "data analytics", "data collection", "data management", "data mining", "data modeling",
    "data quality", "data science", "data scientist", "data structures", "data visualization",
    "deep learning", "docker", "excel", "financial services", "flask", "forecasting", "git", "hadoop",
    "html", "java", "javascript", "keras", "logistic regression", "machine learning", "management",
    "matplotlib", "natural language processing", "neural networks", "nlp", "numpy", "pandas", "power bi",
    "predictive modeling", "programming", "project management", "python", "pytorch", "r", "react", "sas",
    "scikit", "scipy", "seaborn", "selenium", "spark", "sql", "statistical modeling", "statistics",
    "tableau", "tensorflow", "testing", "web scraping",
    "genai", "langchain", "prompt engineering", "llm", "transformers", "bert", "gpt",
    "reinforcement learning", "stable diffusion", "diffusion models", "rag", "vector databases", "fine-tuning",
    "mlops", "data engineering", "feature engineering", "explainable AI", "federated learning", 
    "auto ml", "graph neural networks", "time series forecasting", "edge computing", "quantum computing",
    "ethical AI", "data privacy", "computer graphics", "opencv", "mlflow", "weights & biases", "wandb", 
    "fastai", "jupyterlab", "streamlit", "hpo", "hyperparameter tuning", "knowledge graphs", "neptune.ai"

]

# Sentence-Transformer (semantic)
model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device("cpu"))
# Precompute skill embeddings once (perf optimization)
SKILL_EMB = model.encode(new_top_skills, convert_to_tensor=True, device=torch.device("cpu"))


IMPACT_VERBS = {
    "improved","reduced","increased","decreased","accelerated","optimized",
    "launched","scaled","automated","saved","cut","boosted"
}

def tokenize_simple(text: str):
    return re.findall(r"[a-zA-Z0-9+#.-]+", (text or "").lower())

def bm25_keyword_coverage(resume_text, jd_text):
    jd_tokens = tokenize_simple(jd_text)
    resume_tokens = tokenize_simple(resume_text)
    if not jd_tokens or not resume_tokens:
        return 0.0
    bm25 = BM25Okapi([resume_tokens])
    jd_unique = list(set(jd_tokens))
    scores = [bm25.get_scores([t])[0] for t in jd_unique]
    s = float(np.mean(scores))
    return min(s / 6.0, 1.0)

def stack_tool_coverage(resume_text, jd_text):
    STACK = ["aws","azure","gcp","docker","kubernetes","tensorflow","pytorch","pandas",
             "numpy","scikit-learn","spark","hadoop","react","flask","django","sql","tableau","power bi"]
    r = (resume_text or "").lower()
    j = (jd_text or "").lower()
    needed = {s for s in STACK if s in j}
    if not needed:
        return 0.0
    matched = {s for s in needed if s in r}
    return len(matched) / len(needed)

def achievement_signal(resume_text):
    r = (resume_text or "").lower()
    verbs = sum(1 for v in IMPACT_VERBS if v in r)
    nums  = len(re.findall(r"(\d+(\.\d+)?%?)", r))
    return min((verbs*0.15 + nums*0.05), 1.0)

def formatting_signal(resume_text):
    r = (resume_text or "").lower()
    pts = 0
    for sec in ["experience","education","skills"]:
        if sec in r: pts += 0.2
    wc = len(r.split())
    if 300 <= wc <= 800:
        pts += 0.4
    return min(pts, 1.0)

def extract_skills_regex(text, reference_skills):
    """Exact matches with word boundaries."""
    text_lower = text.lower()
    found = []
    for skill in reference_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return set(found)

def extract_skills_semantic(text, reference_skills, skill_emb, threshold=0.45):
    """
    Semantic fallback: compare each skill embedding against the WHOLE document embedding.
    Fast & robust vs splitting into tokens.
    """
    text = (text or "").strip()
    if not text:
        return set()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 2]  
    if not sentences:
        return set()
    sent_emb = model.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(skill_emb, sent_emb)  
    max_sims = sims.max(dim=1).values.cpu().numpy()  
    sem_found = {reference_skills[i] for i, s in enumerate(max_sims) if s >= threshold}
    return sem_found

def extract_skills_from_text(text):
    exact = extract_skills_regex(text, new_top_skills)
    semantic = extract_skills_semantic(text, new_top_skills, SKILL_EMB, threshold=0.45)
    return sorted(exact.union(semantic))

def fetch_my_score(resume_text):
    skills = extract_skills_from_text(resume_text)
    count = len(skills)
    if count > 20:
        final_score = 9
    elif 15 <= count < 20:
        final_score = 8
    elif 10 <= count <= 14:
        final_score = 7
    elif 6 <= count <= 9:
        final_score = 6
    elif count < 2:
        final_score = 1
    else:
        final_score = 4
    remaining_skills = [i for i in new_top_skills if i not in skills]
    return final_score, skills, remaining_skills


def preprocess_text(text):
    doc = nlp(text or "")
    return [t.lemma_ for t in doc if not t.is_stop]

def remove_un(text):
    if isinstance(text, str):
        words = text.split()
        cleaned = []
        for w in words:
            token = ''.join(ch for ch in w if ch.isalnum()).lower()
            if token and token not in EN_STOPWORDS:
                cleaned.append(token)
        return ' '.join(cleaned)
    elif isinstance(text, list):
        return [remove_un(t) for t in text if isinstance(t, str)]
    return ""

def extract_jd_experience(jd_text):
    match = re.search(r'(\d+)\s*\+?\s*years?', (jd_text or "").lower())
    return int(match.group(1)) if match else 0

def extract_jd_education(jd_text):
    jd_lower = (jd_text or "").lower()
    if "phd" in jd_lower or "doctorate" in jd_lower:
        return "PhD"
    elif "master" in jd_lower or "m.s." in jd_lower or "msc" in jd_lower:
        return "Masters"
    elif "bachelor" in jd_lower or "b.s." in jd_lower or "bsc" in jd_lower:
        return "Bachelors"
    return None

def extract_resume_education(resume_text):
    resume_lower = (resume_text or "").lower()
    if "phd" in resume_lower or "doctorate" in resume_lower:
        return "PhD"
    elif "master" in resume_lower or "m.s." in resume_lower or "msc" in resume_lower:
        return "Masters"
    elif "bachelor" in resume_lower or "b.s." in resume_lower or "bsc" in resume_lower:
        return "Bachelors"
    return None

def calculate_education_match(resume_edu, jd_edu):
    if not jd_edu:
        return 1.0
    levels = {"Bachelors": 1, "Masters": 2, "PhD": 3}
    resume_level = levels.get(resume_edu, 0)
    jd_level = levels.get(jd_edu, 0)
    return 1.0 if resume_level >= jd_level else 0.0

def calculate_resume_experience(job_ranges):
    if not job_ranges:
        return 0.0
    total = sum(max(0, end - start + 1) for start, end in job_ranges.values())  
    return total

def calculate_semantic_similarity(resume_text, job_desc):
    if not resume_text or not job_desc:
        return 0.0
    r_emb = model.encode(resume_text, convert_to_tensor=True)
    j_emb = model.encode(job_desc, convert_to_tensor=True)
    return float(util.cos_sim(r_emb, j_emb).item())

# ATS Score Function

def calculate_ats_score(resume_text, job_desc, skills, semantic_sim, exp_match, jd_skills, resume_edu, jd_edu):
    WEIGHTS = {
        "keyword_bm25": 20, "skill_match": 30, "semantic_sim": 15, "experience": 15,
        "stack_tools": 8, "education": 7, "achievements": 3, "formatting": 2
    }
    kw = bm25_keyword_coverage(resume_text, job_desc)  
    skill_match = len(set(skills) & set(jd_skills)) / max(1, len(jd_skills))
    edu_match = calculate_education_match(resume_edu, jd_edu)  
    stk = stack_tool_coverage(resume_text, job_desc)  
    ach = achievement_signal(resume_text)  
    fmt = formatting_signal(resume_text)  
    score = (
        WEIGHTS["keyword_bm25"] * kw +
        WEIGHTS["skill_match"] * skill_match +
        WEIGHTS["semantic_sim"] * semantic_sim +
        WEIGHTS["experience"] * exp_match +
        WEIGHTS["stack_tools"] * stk +
        WEIGHTS["education"] * edu_match +
        WEIGHTS["achievements"] * ach +
        WEIGHTS["formatting"] * fmt
    )
    return round(float(np.clip(score, 0, 100)), 2)

def resume_score_gauge(score):
    if score <= 45:
        arc_color = "#E50C0C"; text_color = "#FF4C4C"
    elif score <= 75:
        arc_color = "#F56E06"; text_color = "#BDBD5E"
    else:
        arc_color = "#2ECC71"; text_color = "#2ECC71"
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": text_color}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'showticklabels': False, 'visible': False},
            'bar': {'color': arc_color, 'thickness': 0.18},
            'bgcolor': "white",
            'threshold': {'line': {'color': arc_color, 'width': 0}, 'value': score},
            'steps': [{'range': [score, 100], 'color': "#eee"}]
        }
    ))
    fig.update_layout(
        title="ATS Score",
        margin=dict(t=40, b=0, l=0, r=0),
        height=220,
        font={"color": "black", "family": "Arial"},
        autosize=True
    )
    return fig

def top_n_actual_skills(text, n=10):
    if not text:
        return []
    vect = TfidfVectorizer(stop_words='english', max_features=200)
    X = vect.fit_transform([text])
    features = vect.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).ravel()
    filtered = [f for f in features if f.lower() in new_top_skills]
    feature_scores = {f: scores[i] for i, f in enumerate(features) if f in filtered}
    return sorted(feature_scores, key=feature_scores.get, reverse=True)[:n]

def colored_span(word, color="#ffcccb"):
    return f"<span style='background:{color};padding:4px 12px;margin:4px 4px 4px 0;display:inline-block;border-radius:7px;color:black;font-size:16px;font-weight:500'>{word}</span>"

def skill_similarity_score(resume_skills, jd_skills):
    """
    Calculates skill similarity between resume and JD.
    Returns a percentage score.
    """
    if not jd_skills:
        return 0.0  
    matched_skills = [s for s in resume_skills if s in jd_skills]
    score = len(matched_skills) / len(jd_skills)
    return round(score * 100, 2)  

def parse_years(text):
    text_lower = (text or "").lower()
    job_ranges = {} 
    section_patterns = r'\b(work\s*experience|professional\s*experience|employment\s*history|experience|erfahrung|berufserfahrung)\b'
    m = re.search(section_patterns, text_lower)
    if not m:
        return job_ranges
    section_start = m.end()
    next_sections = ['projects', 'achievements', 'skills', 'education', 'profile links', 'technical skills']
    section_end = len(text)
    for ns in next_sections:
        pos = text_lower.find(ns, section_start)
        if pos != -1 and pos < section_end:
            section_end = pos
            break
    exp_text = text[section_start:section_end]
    lines = [line.strip() for line in exp_text.split("\n") if line.strip()]
    ignore_words = {"education", "bachelor", "master", "university", "college", "school", "degree", "diploma"}
    current_date = datetime.datetime.now()
    current_year = current_date.year
    for line in lines:
        if any(w in line.lower() for w in ignore_words):
            continue
        # Extract potential job title (before dates) - improved to handle more formats
        title_match = re.match(r'^(.*?)(?=\s*(?:\d{1,2}[/.-]\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*\d{4}|20\d{2}|\d{4}))', line, re.IGNORECASE)
        job_title = title_match.group(1).strip() if title_match else line.split(',')[0].strip() or line.split('-')[0].strip()
        job_title = job_title.title()  # Capitalize for display
        if not job_title or len(job_title) < 3:  # Skip very short/invalid titles
            continue
        # Use datefinder to extract all dates in the line
        dates = list(datefinder.find_dates(line))
        dates = [d for d in dates if 1900 <= d.year <= current_year + 5]  # Filter reasonable years
        if len(dates) >= 2:
            start_date = min(dates)
            end_date = max(dates)
        elif len(dates) == 1:
            start_date = dates[0]
            end_date = start_date 
        else:
            years = re.findall(r'\b(19[8-9]\d|20[0-3]\d)\b', line)
            years = [int(y) for y in years if 1900 <= int(y) <= current_year + 5]
            if len(years) >= 2:
                start_date = datetime.datetime(min(years), 1, 1)
                end_date = datetime.datetime(max(years), 12, 31)
            elif len(years) == 1:
                start_date = datetime.datetime(years[0], 1, 1)
                end_date = datetime.datetime(years[0], 12, 31)
            else:
                continue
        # Handle 'present' or 'current'
        if re.search(r'\b(present|current|ongoing)\b', line.lower()):
            end_date = current_date
        start_year = start_date.year
        end_year = end_date.year
        if start_year <= end_year:
            if job_title in job_ranges:
                # Merge if duplicate
                existing_start, existing_end = job_ranges[job_title]
                job_ranges[job_title] = (min(existing_start, start_year), max(existing_end, end_year))
            else:
                job_ranges[job_title] = (start_year, end_year)
    return job_ranges

def draw_timeline(job_ranges):
    if not job_ranges:
        st.info("No work experience details found to build chart.")
        return
    # Sort jobs by start year for better ordering
    sorted_jobs = sorted(job_ranges.items(), key=lambda x: x[1][0])
    job_titles = [job for job, _ in sorted_jobs]
    years_duration = [end - start + 1 for _, (start, end) in sorted_jobs]
    
    # Calculate total experience
    total_experience = sum(years_duration)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#3366CC', '#99CC33', '#CC3333']  # Colors from your image (blue, green, red)
    bars = ax.bar(job_titles, years_duration, color=[colors[i % len(colors)] for i in range(len(job_titles))], edgecolor='black')
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} yr',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    ax.set_xlabel("Job Title")
    ax.set_ylabel("Years")
    ax.set_title(f"Experience Duration by Job Title (Total: {total_experience} years)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)

BUZZWORDS = ["responsible for","hardworking","team player","detail oriented","self motivated","go getter",
             "excellent communication","excellent interpersonal","works well","excellent problem solver"]
ACTION_SUGGESTIONS = {
    "responsible for":"Use action verbs: Developed, Implemented, Led",
    "hardworking":"Replace with impact: Achieved X by doing Y",
    "team player":"Prefer: Collaborated with X to achieve Y"}

def buzzword_check(text):
    found = [b for b in BUZZWORDS if b in (text or "").lower()]
    suggestions = [ACTION_SUGGESTIONS.get(b, "Use stronger verb") for b in found]
    return list(zip(found, suggestions))

def generate_resume_tips(resume_text, ats_score, mentioned_skills, num_pages):
    tips, positives = [], []
    if num_pages > 1:
        tips.append("**Keep it to 1 Page**: Aim 400–600 words for better ATS parsing.")
    else:
        positives.append("Length is ATS-friendly (1 page).")
    lower = (resume_text or "").lower()
    missing_sections = []
    if "experience" not in lower: missing_sections.append("'Professional Experience'")
    if "education" not in lower:  missing_sections.append("'Education'")
    if missing_sections:
        tips.append(f"**Add clear sections**: {', '.join(missing_sections)} with quantified impact.")
    else:
        positives.append("Experience/Education sections are present and clear.")
    word_count = len((resume_text or "").split())
    if len(mentioned_skills) < 5 or word_count < 150:
        tips.append("**Show skills in context** (projects/impact) not just in a list.")
    else:
        positives.append("Skills are well-integrated into achievements.")
    if ats_score < 50:
        tips.append("**Boost ATS**: add JD-specific keywords naturally; avoid tables/images.")
    elif ats_score >= 75:
        positives.append(f"Strong ATS score ({ats_score}). Keep tailoring to JD.")
    if tips:
        tips.append("**Quick win**: Use standard fonts (Arial/Calibri 10–12), and spell-check.")
    return tips, positives


def generate_csv_report(data_rows):
    df = pd.DataFrame(data_rows)
    return df.to_csv(index=False)

def generate_pdf_report(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for k, v in data_dict.items():
        text = f"{k}: {v}"
        pdf.multi_cell(0, 8, txt=text)
    return pdf.output(dest='S').encode('latin-1')  # return bytes


# Instructions Card (extracted and added)

def instructions_card():
    st.info(
        """**Quick Start Guide**
1. **Upload Resumes (PDF only)**  
   - Upload one or more resumes (max 10MB each). Ensure text is extractable (not scanned images).  
2. **Paste the Job Description**  
   - Enter the job description in the sidebar for better skill and keyword matching.  
3. **What You’ll See After Upload**  
   - ✅ **Skills Detected**: Technical skills found in the resume.  
   - 📊 **ATS Score (/100)**: Color-coded gauge showing resume-job fit.  
   - 🧠 **Semantic Similarity**: How closely the resume matches the JD.  
   - 🧩 **Experience Timeline**: Visual of work experience years.  
   - ⚠️ **Buzzwords & Suggestions**: Tips to replace overused phrases.  
   - 💡 **Resume Improvement Tips**: Actionable advice to boost your resume.
---
💡 *This tool helps you tailor your resume by analyzing skills, keywords, and structure.*  
""",
        icon="👋",
    )


# Streamlit App

def main():
    st.title("AI Driven Resume Screening and ATS Scoring")
    # Initialize session state
    if "jd_entered" not in st.session_state:
        st.session_state.jd_entered = False
    if "resume_uploaded" not in st.session_state:
        st.session_state.resume_uploaded = False
    # Display instructions only if no resume has been uploaded
    if not st.session_state.resume_uploaded:
        instructions_card()
    with st.sidebar:
        st.header("Job Description")
        # Show tips only if no resume uploaded and JD not entered
        if not st.session_state.resume_uploaded and not st.session_state.jd_entered:
            st.info(
                """
                **Tips for Job Description:**
                - Paste the full job description, including required skills and experience.
                - Clear and detailed text improves skill matching and ATS scoring.
                - Check "Skills Not Mentioned" below for resume improvement ideas.
                """
            )
        
        job_description = st.text_area("Paste Job Description", height=300, key="jd_area")
        
        # Check if JD was entered and not just empty/whitespace
        if not st.session_state.jd_entered and job_description.strip():
            st.session_state.jd_entered = True  # After paste + Enter
        
        # Show quick start guide only if no resume uploaded, JD entered, and no resume uploaded
        if not st.session_state.resume_uploaded and st.session_state.jd_entered:
            st.markdown("""
            **Quick Start Guide**
            1. **Upload Resumes (PDF only)**
                - Upload one or more resumes (max 10MB each). Ensure text is extractable (not scanned images).
            2. **Paste the Job Description** (already done)
            3. **What You'll See After Upload**
                - **Skills Detected**
                - **ATS Score (/100)**
                - **Semantic Similarity**
                - **Experience Timeline**
                - **Buzzwords & Suggestions**
                - **Resume Improvement Tips**
            *This tool helps you tailor your resume by analyzing skills, keywords, and structure.*
            """)

    uploaded_files = st.file_uploader("Upload resumes (PDF)", type="pdf", accept_multiple_files=True, key="resume_upload")
    if uploaded_files and not st.session_state.resume_uploaded:
        st.session_state.resume_uploaded = True
        st.rerun()  # Force a rerun to immediately hide instructions

    if uploaded_files:
        # Parse all resumes
        resumes = []
        for f in uploaded_files:
            try:
                reader = PdfReader(f)
                pages = reader.pages
                text = "".join([p.extract_text() or "" for p in pages])
                resumes.append({"name": f.name, "text": text, "num_pages": len(pages)})
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")

        results = []
        jd_req_years = extract_jd_experience(job_description)
        jd_req_edu = extract_jd_education(job_description)
        all_jd_skills = extract_skills_from_text(job_description or "")  
        for res in resumes:
            resume_text = res["text"]
            num_pages = res["num_pages"]
            score9, mentioned_skills, not_mentioned_skills = fetch_my_score(resume_text)
            semantic_similarity = calculate_semantic_similarity(resume_text, job_description)
            job_ranges = parse_years(resume_text)
            resume_exp_years = calculate_resume_experience(job_ranges)
            exp_match = 1.0 if jd_req_years == 0 else min(resume_exp_years / jd_req_years, 1.0)
            experience_flag = exp_match < 1.0 and jd_req_years > 0
            resume_edu = extract_resume_education(resume_text)
            edu_match = calculate_education_match(resume_edu, jd_req_edu)
            education_flag = edu_match < 1.0 and jd_req_edu is not None
            ats_score = calculate_ats_score(
                resume_text=resume_text,
                job_desc=job_description,
                skills=mentioned_skills,
                semantic_sim=semantic_similarity,
                exp_match=exp_match,
                jd_skills=all_jd_skills,
                resume_edu=resume_edu,
                jd_edu=jd_req_edu
            )

            results.append({
                "name": res["name"],
                "resume_text": resume_text,
                "num_pages": num_pages,
                "score9": score9,  
                "ats_score": ats_score,
                "semantic_similarity": semantic_similarity,
                "experience_match": exp_match,
                "experience_flag": experience_flag,
                "education_flag": education_flag,
                "mentioned_skills": mentioned_skills,
                "not_mentioned_skills": not_mentioned_skills,
                "job_ranges": job_ranges,
                "resume_exp_years": resume_exp_years
            })
        # Rank by ATS
        results.sort(key=lambda x: x["ats_score"], reverse=True)

        with st.sidebar:
            if results:
                st.header("Skills Not Mentioned (Recommendations)")
                for r in results:
                    st.subheader(r["name"])
                    if r["not_mentioned_skills"]:
                        for idx, skill in enumerate(r["not_mentioned_skills"]):
                            st.write(f"{idx} : \"{skill}\"")
                    else:
                        st.write("All reference skills are mentioned — excellent!")
        st.header("Resume Analysis Dashboard")

        for idx, r in enumerate(results):
            with st.expander(f"Resume: {r['name']} (Rank {idx+1})", expanded=(idx == 0)):
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col2.metric("ATS Score", f"{r['ats_score']}/100")
                col3.metric("Semantic Similarity", f"{r['semantic_similarity']:.2f}")
                # Gauge
                st.plotly_chart(resume_score_gauge(r["ats_score"]), use_container_width=True)

                # Experience check
                st.markdown(
                    f"<span style='color: #FFFFE0;'>Your ATS score is {r['ats_score']}/100. indicating that there’s potential to further refine your resume {'best fit' if r['ats_score'] > 80 else 'good' if r['ats_score'] > 60 else 'need improvement'}. Scores above 80 are considered excellent, while those below 60 present opportunities to enhance alignment with ATS standards and boost your chances of making a strong impression.</span>",
                    unsafe_allow_html=True)
                if r["experience_flag"]:
                    if r["resume_exp_years"] == 0:
                        st.markdown(
                            "<span style='color:orange'>Fresher Candidate:</span> No prior experience detected.",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<span style='color:red'>Experience Mismatch:</span> Resume experience appears less than JD requirement.",
                            unsafe_allow_html=True)
                else:
                    if jd_req_years > 0:
                        st.markdown(
                            "<span style='color:green'>Experience Match:</span> Resume experience meets or exceeds JD requirement.",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<span style='color:green'>No Experience Requirement:</span> Specified in JD.",
                            unsafe_allow_html=True)

                # Education check
                if r["education_flag"]:
                    st.markdown(
                        "<span style='color:red'>Education Mismatch:</span> Resume education appears lower than JD requirement.",
                        unsafe_allow_html=True)
                else:
                    if jd_req_edu is not None:
                        st.markdown(
                            "<span style='color:green'>Education Match:</span> Resume education meets or exceeds JD requirement.",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<span style='color:green'>No Education Requirement:</span> Specified in JD.",
                            unsafe_allow_html=True)
                
                # Skills
                st.subheader("Skills Mentioned in Resume")
                st.markdown("".join([colored_span(s, "#A2D5AB") for s in r["mentioned_skills"]]) or "—", unsafe_allow_html=True)


                st.subheader("Skills Mentioned in JD")
                if all_jd_skills:
                    st.markdown(
                        "".join([
                            colored_span(k, "#A2D5AB") if k in r["mentioned_skills"] else colored_span(k, "#FFB3B3")
                            for k in all_jd_skills
                        ]),
                        unsafe_allow_html=True
                    )
                    st.markdown("""
Note: Skills highlighted in <span style='color:#A2D5AB; font-weight:bold;'>green</span> are already included in your resume, 
while those in <span style='color:#FFB3B3; font-weight:bold;'>red</span> are relevant to the job description but currently missing. 
Adding or improving these missing skills can strengthen your resume and increase your chances of getting the job.
""", unsafe_allow_html=True)
                else:
                    st.write("No technical skills found in JD.")

                # Skill Similarity Score
                skill_score = skill_similarity_score(r["mentioned_skills"], all_jd_skills)
                st.markdown(f"<p style='font-size:24px;'><b>Skill Similarity Score:</b> {skill_score}%</p>", unsafe_allow_html=True)

                # Timeline
                st.subheader("Experience Timeline")
                draw_timeline(r["job_ranges"])

                # Buzzword detector
                st.subheader("Buzzword Detector & Suggestions")
                buzz = buzzword_check(r["resume_text"])
                if buzz:
                    for b, sugg in buzz:
                        st.write(f"{colored_span(b,'#FFD2D2')} → Suggestion: {sugg}", unsafe_allow_html=True)
                else:
                    st.write("No common buzzwords found — good!")

                # Tips
                st.subheader("Resume Improvement Tips")
                tips, positives = generate_resume_tips(r["resume_text"], r["ats_score"], r["mentioned_skills"], r["num_pages"])
                if tips:
                    st.markdown("**Areas to Improve:**")
                    for t in tips:
                        st.markdown(f"- {t}")
                if positives:
                    st.markdown("**What's Working Well:**")
                    for p in positives:
                        st.markdown(f"- {p}")
                if not tips and not positives:
                    st.write("Your resume looks strong overall—keep up the great work!")
        st.markdown("---")

if __name__ == "__main__":
    main()











