import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import json
import os

# Use token from secrets (free HF token required for reliable Mistral)
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Add your free Hugging Face token in Streamlit Secrets as 'HF_TOKEN' for reliable API access. Get at huggingface.co/settings/tokens")
    st.stop()

client = InferenceClient(token=hf_token)

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() or "No text extracted."
    except Exception as e:
        return f"PDF error: {str(e)}"

def screen_resumes(jd_text, resume_texts):
    prompt_start = """
    [INST] You are a fair recruiter screening blindly (no names/gender/age bias).
    Job Description: 
    """
    prompt_jd = jd_text.strip()
    
    prompt_middle = """
    
    Output ONLY valid JSON array:
    [{"resume_id": 1, "overall_score": 0-100, "explanation": "2-3 sentences fit", "strengths": ["item1"], "gaps": ["item1"]}, ...]
    
    Resumes:
    """
    
    formatted_resumes = ""
    for i, text in enumerate(resume_texts):
        formatted_resumes += f"Resume {i+1} (truncated):\n{text[:2500]}\n\n"  # Shorter for API limits
    
    prompt_end = "[/INST]"
    
    full_prompt = f"{prompt_start}{prompt_jd}{prompt_middle}{formatted_resumes}{prompt_end}"
    
    with st.spinner("Screening via Hugging Face (Mistral-7B)..."):
        try:
            raw_output = client.text_generation(
                full_prompt,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_new_tokens=1000,
                temperature=0.3
            )
        except Exception as e:
            return [{"error": f"API issue: {str(e)} – Try shorter JD/resume or check token"}]
    
    try:
        start = raw_output.find("[")
        if start == -1:
            raise ValueError("No JSON found")
        end = raw_output.rfind("]") + 1
        json_str = raw_output[start:end]
        results = json.loads(json_str)
    except Exception as e:
        results = [{"error": f"Parse issue: {str(e)}", "raw": raw_output[:800]}]
    
    return results

# UI
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🚀 AI Resume Screener App")
st.markdown("Fair, explainable LLM screening – by Owadokun Tosin Tobi")

jd_text = st.text_area("Job Description (required)", height=150, placeholder="Paste JD...")

uploaded_resumes = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if st.button("🔍 Screen Resumes", type="primary"):
    if jd_text.strip() and uploaded_resumes:
        with st.spinner("Extracting PDFs..."):
            resume_texts = [extract_text_from_pdf(f) for f in uploaded_resumes]
        
        results = screen_resumes(jd_text, resume_texts)
        
        st.success("Screening Complete!")
        
        if "error" not in results[0]:
            results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        for res in results:
            if "error" in res:
                st.error("Issue:")
                if "raw" in res:
                    st.code(res["raw"])
                else:
                    st.write(res["error"])
            else:
                st.markdown(f"### Resume {res['resume_id']} – **Score: {res['overall_score']}/100**")
                st.write("**Explanation**: " + res["explanation"])
                st.write("**Strengths**: " + ", ".join(res["strengths"]))
                st.write("**Gaps**: " + ", ".join(res["gaps"]))
                st.divider()
    else:
        st.warning("Add JD + resumes.")

st.caption("Hugging Face Mistral-7B (free tier with token) – Dec 16, 2025. Your resume PDF tested great!")