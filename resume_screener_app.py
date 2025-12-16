import streamlit as st
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
import json

# Optional token for better rate limits (add in Secrets as HF_TOKEN if you have one)
hf_token = st.secrets.get("HF_TOKEN")

client = InferenceClient(token=hf_token)  # Works without token too for this model

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
        formatted_resumes += f"Resume {i+1} (truncated):\n{text[:2500]}\n\n"
    
    prompt_end = "[/INST]"
    
    full_prompt = f"{prompt_start}{prompt_jd}{prompt_middle}{formatted_resumes}{prompt_end}"
    
    with st.spinner("Screening via Hugging Face (Zephyr-7B)..."):
        try:
            raw_output = client.text_generation(
                full_prompt,
                model="HuggingFaceH4/zephyr-7b-beta",
                max_new_tokens=1000,
                temperature=0.3,
                return_full_text=False  # Only new tokens
            )
        except Exception as e:
            return [{"error": f"API issue: {str(e)} – Try shorter input"}]
    
    try:
        start = raw_output.find("[")
        if start == -1:
            raise ValueError("No JSON")
        end = raw_output.rfind("]") + 1
        json_str = raw_output[start:end]
        results = json.loads(json_str)
    except Exception as e:
        results = [{"error": f"Parse issue: {str(e)}", "raw": raw_output[:800]}]
    
    return results

# UI (same)
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🚀 AI Resume Screener App")
st.markdown("Fair, explainable LLM screening – by Owadokun Tosin Tobi")

jd_text = st.text_area("Job Description (required)", height=150)

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

st.caption("Zephyr-7B (free tier) – Great for JSON output. Your resume PDF extracts perfectly!")