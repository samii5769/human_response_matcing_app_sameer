import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import pdfplumber

# --- PAGE SETUP ---
st.set_page_config(page_title="Human Pair matching", layout="wide")
st.title(" Human pair Matching Algorithm")
st.markdown("Upload your human responsce datasets for pair matching.")

# --- SIDEBAR: WEIGHT ADJUSTMENTS ---
st.sidebar.header("⚙️ Adjust Algorithm Weights")
st.sidebar.markdown("**Hard Skills (Default 40%)**")
w_spec = st.sidebar.slider("Specialisation Weight", 0.0, 1.0, 0.25)
w_deg = st.sidebar.slider("Degree Weight", 0.0, 1.0, 0.15)

st.sidebar.markdown("**Soft Skills (Default 60%)**")
w_prof = st.sidebar.slider("Professional Match", 0.0, 1.0, 0.20)
w_pers = st.sidebar.slider("Personal Fit", 0.0, 1.0, 0.15)
w_iit = st.sidebar.slider("IIT Context", 0.0, 1.0, 0.15)
w_back = st.sidebar.slider("Background & Values", 0.0, 1.0, 0.10)

st.sidebar.markdown("**Bonuses**")
bonus_female = st.sidebar.slider("Female-to-Female Bonus", 0.0, 0.5, 0.10)

# --- HELPER FUNCTIONS ---
def clean(text): return str(text).lower().strip() if pd.notnull(text) else ""

def get_degree_group(text):
    t = clean(text)
    if any(x in t for x in ['b.tech', 'btech', 'undergraduate', 'bachelor']): return 1
    if any(x in t for x in ['dual degree', 'bs-ms', 'iddd']): return 2
    if any(x in t for x in ['masters', 'm.tech', 'msc']): return 3
    if any(x in t for x in ['mba', 'emba']): return 4
    if any(x in t for x in ['phd', 'doctorate']): return 5
    return 0

c_branch_map = {'engineering design': 1, 'biotechnology': 2, 'biological': 2, 'civil': 3, 'physics': 4, 'mechanical': 5, 'applied mechanics': 5, 'chemical': 6, 'computer science': 7, 'data science': 7, 'electrical': 8, 'electronics': 8, 'metallurgical': 9, 'aerospace': 10, 'management': 11, 'naval': 12, 'ocean': 12, 'mathematics': 13, 'humanities': 14}
m_spec_map = {'microbiology': 1, 'bio': 1, 'physics': 2, 'civil': 3, 'mechanical': 4, 'chemical': 5, 'computer': 6, 'cs': 6, 'electrical': 7, 'electronics': 7, 'metallurgical': 8, 'aeronautical': 9, 'management': 10, 'finance': 10, 'naval': 11, 'math': 12}
spec_match_logic = {1: [4], 2: [1], 3: [3], 4: [2], 5: [4], 6: [5], 7: [6], 8: [7], 9: [8], 10: [9], 11: [10], 12: [11], 13: [12], 14: []}

def get_group(text, mapping):
    t = clean(text)
    for k, v in mapping.items():
        if k in t: return v
    return 0

# --- MAIN UI: FILE UPLOADS ---
col1, col2 = st.columns(2)
with col1:
    coachee_file = st.file_uploader("Upload Coachee Data", type=['csv', 'xlsx', 'pdf'])
with col2:
    mentor_file = st.file_uploader("Upload Mentor Data", type=['csv', 'xlsx', 'pdf'])

#  Detects file type and loads it into a Pandas DataFrame
def load_data(file):
    file_name = file.name.lower()
    
    try:
        if file_name.endswith('.csv'):
            # First attempt: standard UTF-8 encoding
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                # If it fails, reset the file pointer to the beginning and try Latin-1
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
                
        elif file_name.endswith('.xlsx'):
            # Requires 'openpyxl' engine
            return pd.read_excel(file, engine='openpyxl')
            
        elif file_name.endswith('.pdf'):
            # Extract tables from PDF
            with pdfplumber.open(file) as pdf:
                all_rows = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_rows.extend(table)
                
                if all_rows:
                    # Assume the first row extracted is the header
                    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
                    # Clean up any purely empty rows/columns from PDF extraction artifacts
                    df.dropna(how='all', inplace=True)
                    return df
                else:
                    st.error(f"⚠️ Could not find a readable table in {file.name}.")
                    return None
                    
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None
if coachee_file and mentor_file:
    if st.button("Run Matching Algorithm", type="primary"):
        with st.spinner("Analyzing text and calculating optimal matches..."):
            
            # --- NEW DATA LOADING ---
            coachee_df = load_data(coachee_file)
            mentor_df = load_data(mentor_file)
            
            # Check if data loaded successfully before proceeding
            if coachee_df is None or mentor_df is None:
                st.stop() # Stops execution if a file couldn't be read
            
            # Load Data
            coachee_df = load_data(coachee_file)
            mentor_df = load_data(mentor_file)
            
            # Stop the app from crashing if the data couldn't be loaded
            if coachee_df is None or mentor_df is None:
                st.stop()
            
            # Preprocess
            coachee_df['Batch'] = coachee_df['Map Code/Coachee mapping'].astype(str).apply(lambda x: x.split('-')[1] if '-' in x else '0')
            mentor_df['Batch'] = mentor_df['Mentor ID'].astype(str).apply(lambda x: x.split('-')[1] if '-' in x else '0')
            coachee_df['Deg_Grp'] = coachee_df['Program at IIT Madras'].apply(get_degree_group)
            mentor_df['Deg_Grp'] = mentor_df['Degree'].apply(get_degree_group)
            coachee_df['Branch_Grp'] = coachee_df['Branch at IIT Madras'].apply(lambda x: get_group(x, c_branch_map))
            mentor_df['Spec_Grp'] = mentor_df['Specialisation'].apply(lambda x: get_group(x, m_spec_map))
            
            # Combine Text
            def combine(df, cols): return df[cols].fillna('').apply(lambda x: ' '.join(x), axis=1).apply(clean)
            
            coachee_df['Txt_Prof'] = coachee_df['Career plan'].apply(clean)
            mentor_df['Txt_Prof'] = mentor_df['Career snapshot'].apply(clean)
            coachee_df['Txt_Pers'] = combine(coachee_df, ['Top 3 interests', 'Main passions'])
            mentor_df['Txt_Pers'] = mentor_df['Interests'].apply(clean)
            coachee_df['Txt_IIT'] = combine(coachee_df, ['IIT trajectory', 'Career plan'])
            mentor_df['Txt_IIT'] = mentor_df['IIT experience'].apply(clean)
            coachee_df['Txt_Back'] = combine(coachee_df, ['Family info and schooling', 'Roll Models'])
            mentor_df['Txt_Back'] = mentor_df['Growing up years'].apply(clean)
            
            # Vectorize
            vec_prof = TfidfVectorizer(stop_words='english').fit(pd.concat([coachee_df['Txt_Prof'], mentor_df['Txt_Prof']]))
            vec_pers = TfidfVectorizer(stop_words='english').fit(pd.concat([coachee_df['Txt_Pers'], mentor_df['Txt_Pers']]))
            vec_iit = TfidfVectorizer(stop_words='english').fit(pd.concat([coachee_df['Txt_IIT'], mentor_df['Txt_IIT']]))
            vec_back = TfidfVectorizer(stop_words='english').fit(pd.concat([coachee_df['Txt_Back'], mentor_df['Txt_Back']]))

            final_matches = []
            def normalize(arr):
                if len(arr) < 2 or arr.max() == 0: return arr
                return (arr - arr.min()) / (arr.max() - arr.min())

            for idx, c_row in coachee_df.iterrows():
                c_id = c_row['Map Code/Coachee mapping']
                batch = c_row['Batch']
                c_gender = clean(c_row['Gender '] if 'Gender ' in c_row else c_row['Gender'])
                
                candidates = mentor_df[mentor_df['Batch'] == batch].copy()
                if candidates.empty: continue
                
                s_prof = normalize(cosine_similarity(vec_prof.transform([c_row['Txt_Prof']]), vec_prof.transform(candidates['Txt_Prof'])).flatten())
                s_pers = normalize(cosine_similarity(vec_pers.transform([c_row['Txt_Pers']]), vec_pers.transform(candidates['Txt_Pers'])).flatten())
                s_iit  = normalize(cosine_similarity(vec_iit.transform([c_row['Txt_IIT']]), vec_iit.transform(candidates['Txt_IIT'])).flatten())
                s_back = normalize(cosine_similarity(vec_back.transform([c_row['Txt_Back']]), vec_back.transform(candidates['Txt_Back'])).flatten())

                scores = []
                for i, (_, m_row) in enumerate(candidates.iterrows()):
                    sc_spec = 1.0 if (c_row['Branch_Grp'] in spec_match_logic and m_row['Spec_Grp'] in spec_match_logic[c_row['Branch_Grp']]) else 0.0
                    sc_deg = 1.0 if (c_row['Deg_Grp'] == m_row['Deg_Grp'] and c_row['Deg_Grp'] != 0) or ((c_row['Deg_Grp']==1 and m_row['Deg_Grp']==2) or (c_row['Deg_Grp']==2 and m_row['Deg_Grp']==1)) else 0.0
                    
                    total = (sc_spec * w_spec) + (sc_deg * w_deg) + (s_prof[i] * w_prof) + (s_pers[i] * w_pers) + (s_iit[i] * w_iit) + (s_back[i] * w_back)
                    if 'female' in c_gender and 'female' in clean(m_row['Gender']): total += bonus_female
                    
                    scores.append({'id': m_row['Mentor ID'], 'score': total, 'details': f"Score: {total:.2f}"})
                
                scores.sort(key=lambda x: x['score'], reverse=True)
                top3, seen = [], set()
                for s in scores:
                    if s['id'] not in seen:
                        top3.append(s); seen.add(s['id'])
                    if len(top3) == 3: break
                
                row = {'Coachee Code': c_id}
                for k in range(3):
                    if k < len(top3):
                        row[f'Option {k+1} ID'] = top3[k]['id']
                        row[f'Option {k+1} Score'] = round(top3[k]['score']*100, 1)
                    else:
                        row[f'Option {k+1} ID'] = "N/A"
                final_matches.append(row)

            res_df = pd.DataFrame(final_matches)
            st.success("Matching Complete!")
            st.dataframe(res_df.head(10))
            
            # Download Button
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Final Matches CSV", data=csv, file_name="Matched_Results.csv", mime="text/csv")
