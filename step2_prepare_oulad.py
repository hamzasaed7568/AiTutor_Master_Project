import pandas as pd
import os

RAW_DIR = "dataset/raw_dataset/oulad"
PROCESSED_DIR = "dataset/processed_dataset/oulad"

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Step 1: OULAD Raw CSV files load ho rahi hain...")
student_assessment = pd.read_csv(os.path.join(RAW_DIR, 'studentAssessment.csv'))
assessments = pd.read_csv(os.path.join(RAW_DIR, 'assessments.csv'))

print("Step 2: Data ko merge aur clean kiya ja raha hai...")
df = pd.merge(student_assessment, assessments, on='id_assessment', how='left')

df = df.dropna(subset=['score'])

print("Step 3: Assessments map with Computer Science Topics...")
topic_mapping = {
    25340: "Algorithms and Logic",
    15020: "Data Structures",
    25360: "Programming Languages",
    25339: "Software Engineering",
    25353: "Operating Systems Architecture"
}

df['cs_topic'] = df['id_assessment'].map(topic_mapping).fillna("General Computing Fundamentals")

df['is_correct'] = (df['score'] >= 40).astype(int)

final_df = df[['id_student', 'id_assessment', 'cs_topic', 'score', 'is_correct']]

print("Step 4: Processed Data save ho raha hai...")
processed_file_path = os.path.join(PROCESSED_DIR, 'cleaned_student_performance.csv')
final_df.to_csv(processed_file_path, index=False)

print(f"\n🎉 Success! Cleaned dataset saved: {processed_file_path}")
print(f"Total student records processed: {len(final_df)}")