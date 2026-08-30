import pandas as pd
import os

# 1. Paths set karein
RAW_DIR = "dataset/raw_dataset/oulad"
PROCESSED_DIR = "dataset/processed_dataset/oulad"

# Agar processed_dataset ka folder nahi bana hua, toh yeh automatically bana dega
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Step 1: OULAD Raw CSV files load ho rahi hain...")
student_assessment = pd.read_csv(os.path.join(RAW_DIR, 'studentAssessment.csv'))
assessments = pd.read_csv(os.path.join(RAW_DIR, 'assessments.csv'))

print("Step 2: Data ko merge aur clean kiya ja raha hai...")
# Assessments aur unke scores ko aapas mein mila rahe hain
df = pd.merge(student_assessment, assessments, on='id_assessment', how='left')

# Jin bacchon ne test submit hi nahi kiya (missing scores), unko data se nikal rahe hain
df = df.dropna(subset=['score'])

print("Step 3: Assessments ko Computer Science Topics ke sath map kar rahe hain...")
# OULAD data ko book ke chapters/topics se link karne ki dictionary (Research assumption for prototype)
# Hum kuch typical assessment IDs ko CS topics ka naam de rahe hain
topic_mapping = {
    25340: "Algorithms and Logic",
    15020: "Data Structures",
    25360: "Programming Languages",
    25339: "Software Engineering",
    25353: "Operating Systems Architecture"
}

# Naya column 'cs_topic' bana rahe hain
df['cs_topic'] = df['id_assessment'].map(topic_mapping).fillna("General Computing Fundamentals")

# BKT (Bayesian Knowledge Tracing) model ke liye ek 'is_correct' (Pass/Fail) column zaroori hai. 
# OULAD mein 40 se upar marks pass hote hain. (1 = Pass, 0 = Fail/Weak)
df['is_correct'] = (df['score'] >= 40).astype(int)

# Sirf woh columns filter kar rahe hain jo hamare BKT aur RAG system ko chahiye
final_df = df[['id_student', 'id_assessment', 'cs_topic', 'score', 'is_correct']]

print("Step 4: Processed Data save ho raha hai...")
processed_file_path = os.path.join(PROCESSED_DIR, 'cleaned_student_performance.csv')
final_df.to_csv(processed_file_path, index=False)

print(f"\n🎉 Success! Cleaned dataset yahan save ho gaya hai: {processed_file_path}")
print(f"Total student records processed: {len(final_df)}")