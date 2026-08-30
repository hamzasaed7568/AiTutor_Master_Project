import os
import pandas as pd
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# =========================================================
# STEP 1: SYSTEM LOAD (Vector DB & Local LLM)
# =========================================================
print("Step 1: System Load ho raha hai...")

# Embedding Model (Dimaag ka search engine)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Local Open-Source LLM Load (TinyLlama - Perfect for text-generation)
print("Local LLM (TinyLlama) loading:- ")
pipe = pipeline(
    task="text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=pipe)

# =========================================================
# STEP 2: LOAD OULAD DATA
# =========================================================
print("\nStep 2: getting weak topic of student from CSV...")
df = pd.read_csv("dataset/processed_dataset/oulad/cleaned_student_performance.csv")
weak_student = df[df['is_correct'] == 0].iloc[0]
weak_topic = weak_student['cs_topic']

print(f"Student ID {weak_student['id_student']} weak topic is  '{weak_topic}'.")

# =========================================================
# STEP 3: RAG RETRIEVAL (From OpenStax Book)
# =========================================================
print(f"\nStep 3: '{weak_topic}' searching data related to weak topic...")
retrieved_docs = vector_db.similarity_search(weak_topic, k=2)
context = "\n".join([doc.page_content for doc in retrieved_docs])

# =========================================================
# STEP 4: XAI GENERATION (Explainable AI)
# =========================================================
print("\nStep 4: LLM se Explainable Recommendation generate karwa rahe hain...")

# TinyLlama ke liye special Prompt Format
prompt_template = """<|system|>
You are a helpful AI Computer Science Tutor. Answer briefly and simply.</s>
<|user|>
A student is weak in the topic: {topic}. 
Based on this textbook material:
{context}

Explain in 2 simple sentences why the student should read this material.</s>
<|assistant|>
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "context"])
final_prompt = prompt.format(topic=weak_topic, context=context)

# LLM Generation
response = llm.invoke(final_prompt)

# Clean the response to only show the assistant's output
if "<|assistant|>" in response:
    response = response.split("<|assistant|>")[-1].strip()

print("\n" + "="*60)
print("🎯 AI TUTOR'S RECOMMENDATION:")
print("="*60)
print(response.strip())
print("="*60)