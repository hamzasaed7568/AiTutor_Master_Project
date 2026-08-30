import os
import time
import pandas as pd
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# =========================================================
# STEP 1: INITIALIZATION
# =========================================================
print("Starting Evaluation Pipeline...")

# Load Data
student_data = pd.read_csv("dataset/processed_dataset/oulad/cleaned_student_performance.csv")
# Get 5 students who failed an assessment for testing
test_students = student_data[student_data['is_correct'] == 0].head(5)

# Load RAG Models
print("Loading Models (ChromaDB & TinyLlama)...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

generator_pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=150,
    temperature=0.3
)
local_llm = HuggingFacePipeline(pipeline=generator_pipe)

PROMPT_TEMPLATE = """<|system|>
You are a helpful AI Computer Science Tutor. Answer briefly and simply.
Explain in 2 simple sentences why the student should read this material.
<|assistant|>
"""
prompt_engine = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["topic", "context"])

# =========================================================
# STEP 2: EVALUATION LOOP
# =========================================================
print("\nRunning Evaluation on 5 Students. Please wait...")

results_list = []

for index, row in test_students.iterrows():
    student_id = row['id_student']
    weak_topic = row['cs_topic']
    
    print(f"Evaluating Student ID: {student_id} | Topic: {weak_topic}...")
    
    # Start Timer
    start_time = time.time()
    
    # Retrieval
    relevant_docs = vector_db.similarity_search(weak_topic, k=2)
    text_context = "".join([doc.page_content for doc in relevant_docs])
    
    # Generation
    structured_prompt = prompt_engine.format(topic=weak_topic, context=text_context)
    full_response = local_llm.invoke(structured_prompt)
    
    if "<|assistant|>" in full_response:
        assistant_reply = full_response.split("<|assistant|>")[-1].strip()
    else:
        assistant_reply = full_response.strip()
        
    # End Timer
    end_time = time.time()
    latency = round(end_time - start_time, 2)
    
    # Calculate simple Faithfulness (Context Overlap Metric)
    # Check how many words in the generated answer actually came from the retrieved book text
    answer_words = set(assistant_reply.lower().split())
    context_words = set(text_context.lower().split())
    
    if len(answer_words) > 0:
        overlap = len(answer_words.intersection(context_words))
        faithfulness_score = round((overlap / len(answer_words)) * 100, 2)
    else:
        faithfulness_score = 0
        
    # Append to results
    results_list.append({
        "Student_ID": student_id,
        "Weak_Topic": weak_topic,
        "Latency_Seconds": latency,
        "Faithfulness_Score_%": faithfulness_score,
        "Generated_Word_Count": len(answer_words)
    })

# =========================================================
# STEP 3: SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results_list)

# Calculate Averages for the final thesis report
avg_latency = results_df['Latency_Seconds'].mean()
avg_faithfulness = results_df['Faithfulness_Score_%'].mean()

print("\n" + "="*50)
print("🎯 EVALUATION COMPLETED")
print("="*50)
print(f"Average Response Time: {avg_latency} seconds")
print(f"Average Faithfulness Score: {avg_faithfulness}%")
print("="*50)

results_df.to_csv("thesis_evaluation_results.csv", index=False)
print("Results saved to 'thesis_evaluation_results.csv'.")