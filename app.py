import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
torch.set_num_threads(4) 
import pandas as pd
from flask import Flask, render_template, request
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

print("Backend is loading...")

student_data = pd.read_csv("dataset/processed_dataset/oulad/cleaned_student_performance.csv")

# Initialize HuggingFace models
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

# TinyLlama Local model setup
print("Local TinyLlama LLM is loading...")
generator_pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=150,
    temperature=0.3
)
local_llm = HuggingFacePipeline(pipeline=generator_pipe)
print("TinyLlama loaded!")

# Define special Prompt Template for XAI 
PROMPT_TEMPLATE = """<|system|>
You are a helpful AI Computer Science Tutor. Answer briefly and simply.
Explain in 2 simple sentences why the student should read this material.
<|assistant|>
"""
prompt_engine = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["topic", "context"])

# =========================================================
# STEP 2: HELPER FUNCTIONS (The RAG Pipeline)
# =========================================================

def extract_weakness(student_id):
    """BKT logic: Get the weakest topic from student assessment history"""
    student_records = student_data[student_data['id_student'] == int(student_id)]
    if student_records.empty:
        return None, "Not Found", None
    
    # Filter for failed assessments
    failed_assessments = student_records[student_records['is_correct'] == 0]
    
    # If no failures, provide General Computing Fundamentals topic
    if failed_assessments.empty:
        return student_records.iloc[0]['id_student'], "No weaknesses found. Explore fundamental topics.", None
    
    # Get weak topic from failed assessment
    weakest_topic_info = failed_assessments.iloc[0]
    return weakest_topic_info['id_student'], weakest_topic_info['cs_topic'], weakest_topic_info['id_assessment']

def get_explanation(student_topic):
    """Retrieve material and generate Gemini-like RAG justification"""
    # 1. RETRIEVAL (From Vector DB)
    relevant_docs = vector_db.similarity_search(student_topic, k=2)
    text_context = "".join([doc.page_content for doc in relevant_docs])

    # 2. GENERATION (XAI using LLM)
    structured_prompt = prompt_engine.format(topic=student_topic, context=text_context)
    full_response = local_llm.invoke(structured_prompt)

    # Clean the response to only show assistant output
    if "<|assistant|>" in full_response:
        assistant_reply = full_response.split("<|assistant|>")[-1].strip()
    else:
        assistant_reply = "AI tutor can provide a personalized path for this topic."

    return text_context, assistant_reply

# =========================================================
# STEP 3: FLASK ROUTES
# =========================================================

@app.route('/')
def show_dashboard():
    """Initial Dashboard without result"""
    return render_template('dashboard.html', result=None)

@app.route('/path')
def generate_path():
    """Process student ID and generate customized learning path"""
    # Get student_id from query string
    student_id = request.args.get('student_id')
    
    # If student_id is present, start BKT logic
    if student_id and student_id.isdigit():
        student_id, student_topic, weak_assessment_id = extract_weakness(student_id)

        # If student exists, continue to RAG pipeline
        if student_id:
            textbook_context, ai_justification = get_explanation(student_topic)
            
            final_result = {
                'id_student': student_id,
                'weak_topic': student_topic,
                'id_assessment': weak_assessment_id,
                'justification': ai_justification,
                'book_extract': textbook_context
            }
            return render_template('dashboard.html', result=final_result)
        
        # If student not found
        error_result = {'error': f"Student ID '{request.args.get('student_id')}' not found."}
        return render_template('dashboard.html', result=error_result)

    # Empty search
    return show_dashboard()

if __name__ == '__main__':
    # Start the Flask development server on port 5000
    app.run(debug=True, port=5000, use_reloader=False, threaded=False)