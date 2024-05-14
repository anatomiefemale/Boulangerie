# Import modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a sample dataset of patient records with disease labels
data = {
    'Patient_ID': [1, 2, 3, 4, 5, 6],
    'Notes': [
        "Patient shows increased blood sugar and frequent urination.",
        "Elevated blood pressure and headaches reported repeatedly.",
        "Blood sugar tests indicate possible diabetic condition.",
        "High blood pressure observed, along with blurred vision.",
        "Urine tests confirm high sugar levels, suggesting diabetes.",
        "Patient complains of chronic headaches and high blood pressure."
    ],
    'Disease': ['Diabetes', 'Hypertension', 'Diabetes', 'Hypertension', 'Diabetes', 'Hypertension']
}
df = pd.DataFrame(data)

# Define function to extract top 5 distinguishing keywords using TF-IDF
def extract_distinguishing_keywords(notes):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(notes)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    return keywords_scores[:5]  # Return the top 5 keywords for clarity

# Analyze by disease type
diabetes_df = df[df['Disease'] == 'Diabetes']
hypertension_df = df[df['Disease'] == 'Hypertension']

# Extract keywords for each disease type
diabetes_keywords = extract_distinguishing_keywords(diabetes_df['Notes'])
hypertension_keywords = extract_distinguishing_keywords(hypertension_df['Notes'])

# Print results for each disease
print("Top 5 Keywords for Diabetes:")
for keyword, score in diabetes_keywords:
    print(f"{keyword}: {score:.2f}")

print("\nTop 5 Keywords for Hypertension:")
for keyword, score in hypertension_keywords:
    print(f"{keyword}: {score:.2f}")
