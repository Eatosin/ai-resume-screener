import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import json

client = InferenceClient()  # Free public API

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() or "No text extracted from PDF."
    except Exception as e:
        return f"PDF extraction error: {str(e)}"

def screen_resumes(jd_text, resume_texts):
    prompt_start = """
    [INST] You are a fair, expert recruiter screening blindly (ignore names, gender, age, personal details).
    Job Description: 
    """
    prompt_jd = jd_text
    
    prompt_middle = """
    
    For each resume, output ONLY valid JSON like:
    [{"resume_id": 1, "overall_score": 0-100, "explanation": "2-3 sentences on fit", "strengths": ["item1", "item2"], "gaps": ["item1", "item2"]}, ...]
    
    Resumes:
    """
    
    formatted_resumes = ""
    for i, text in enumerate(resume_texts):
        formatted_resumes += f"Resume {i+1}:\n{text[:3000]}\n\n"  # Truncate safely
    
    prompt_end = "[/INST]"
    
    full_prompt = f"{prompt_start}{prompt_jd}{prompt_middle}{formatted_resumes}{prompt_end}"
    
    with st.spinner("AI screening via Hugging Face API..."):
        try:
            raw_output = client.text_generation(
                full_prompt,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_new_tokens=1200,
                temperature=0.3,
                stop_sequences=["[/INST]"]  # Prevent extra text
            )
        except Exception as e:
            return [{"error": f"API error: {str(e)} (try shorter JD/resumes)"}]
    
    # Robust JSON extract
    try:
        start = raw_output.find("[")
        if start == -1:
            raise ValueError("No JSON array found")
        end = raw_output.rfind("]") + 1
        json_str = raw_output[start:end]
        results = json.loads(json_str)
        if not isinstance(results, list):
            raise ValueError("Not a list")
    except Exception as e:
        results = [{"error": f"Parse failed: {str(e)}", "raw_output": raw_output[:1000]}]  # Truncate raw for display
    
    return results

# UI (unchanged, minor polish)
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🚀 AI Resume Screener App")
st.markdown("Simple, fair, explainable screening powered by LLMs – by Owadokun Tosin Tobi")

jd_text = st.text_area("Job Description (required)", height=150, placeholder="Paste full JD here... e.g., AI Trainer role requiring prompt engineering, content creation...")

uploaded_resumes = st.file_uploader(
    "Upload Resume PDFs (multiple OK)",
    type="pdf",
    accept_multiple_files=True
)

if st.button("🔍 Screen Resumes", type="primary"):
    if jd_text.strip() and uploaded_resumes:
        with st.spinner("Extracting text from PDFs..."):
            resume_texts = [extract_text_from_pdf(f) for f in uploaded_resumes]
            if any("error" in text.lower() for text in resume_texts):
                st.error("PDF read issue—try clearer scans.")
        
        results = screen_resumes(jd_text.strip(), resume_texts)
        
        st.success("Screening Complete!")
        
        if "error" not in results[0]:
            results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        for res in results:
            if "error" in res:
                st.error("Screening Issue:")
                if "raw_output" in res:
                    st.code(res["raw_output"])
                else:
                    st.write(res["error"])
            else:
                st.markdown(f"### Resume {res['resume_id']} – **Score: {res['overall_score']}/100**")
                st.write("**Explanation**: " + res["explanation"])
                st.write("**Strengths**: " + ", ".join(res["strengths"]))
                st.write("**Gaps**: " + ", ".join(res["gaps"]))
                st.divider()
    else:
        st.warning("Add a job description and at least one resume PDF.")

st.caption("Hugging Face API (Mistral-7B) – Free tier may rate-limit heavy use. Your resume PDF works great for testing!")