import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

## 1. Data Loading with Column Verification
def load_and_verify_data(file_path):
    """Load data and verify columns"""
    df = pd.read_csv(file_path)
    print("Available columns:", df.columns.tolist())
    
    # Map expected columns to available columns
    column_mapping = {
        'title': ['Title', 'title', 'Job Title', 'job_title'],
        'company': ['Company', 'company', 'Employer', 'employer'],
        'location': ['Location', 'location', 'City', 'city'],
        'description': ['Description', 'description', 'Job Description', 'job_description'],
        'domain': ['Industry', 'industry', 'Domain', 'domain', 'Category', 'category']
    }
    
    selected_columns = {}
    for target_col, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in df.columns:
                selected_columns[target_col] = col
                break
        else:
            print(f"Warning: No column found for {target_col}, will create empty")
            selected_columns[target_col] = None
    
    return df, selected_columns

# Load data
job_df, column_mapping = load_and_verify_data("Combined_Jobs_Final.csv")
print("\nColumn mapping:", column_mapping)

## 2. Data Preprocessing with Flexible Column Handling
def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def preprocess_data(df, column_mapping):
    """Preprocess the job data with flexible columns"""
    processed_df = pd.DataFrame()
    
    # Handle each column with fallbacks
    for target_col, source_col in column_mapping.items():
        if source_col and source_col in df.columns:
            processed_df[target_col] = df[source_col]
        else:
            processed_df[target_col] = "" if target_col == 'description' else "Unknown"
    
    # Clean text data
    processed_df['cleaned_desc'] = processed_df['description'].apply(clean_text)
    processed_df['cleaned_title'] = processed_df['title'].apply(clean_text)
    
    # Extract skills (simplified approach)
    common_skills = {
        'it': ['python', 'java', 'sql', 'machine learning', 'aws', 'cloud', 'developer', 'software'],
        'healthcare': ['patient', 'medical', 'health', 'care', 'nursing', 'hospital', 'doctor'],
        'finance': ['financial', 'accounting', 'excel', 'analysis', 'banking', 'investment'],
        'marketing': ['marketing', 'social media', 'seo', 'content', 'digital', 'advertising']
    }
    
    def extract_skills(row):
        domain = str(row['domain']).lower()
        desc = row['cleaned_desc']
        skills = []
        
        # Find matching domain
        for dom, skill_list in common_skills.items():
            if dom in domain:
                skills.extend([skill for skill in skill_list if skill in desc])
        
        # Add general skills if no domain specific ones found
        if not skills:
            general_skills = ['communication', 'team', 'management', 'leadership', 'problem solving']
            skills.extend([skill for skill in general_skills if skill in desc])
        
        return list(set(skills))  # Remove duplicates
    
    processed_df['skills'] = processed_df.apply(extract_skills, axis=1)
    
    return processed_df

job_df = preprocess_data(job_df, column_mapping)
print("\nProcessed data sample:")
print(job_df.head())

## 3. Feature Engineering
def create_features(df):
    """Create features for job recommendations"""
    # Combine title and description for better representation
    df['text_features'] = df['cleaned_title'] + ' ' + df['cleaned_desc']
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    
    # Dimensionality reduction with LSA
    svd = TruncatedSVD(n_components=50, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa_features = normalizer.fit_transform(svd.fit_transform(tfidf_matrix))
    
    # Skills vectorization
    mlb = MultiLabelBinarizer()
    skills_matrix = mlb.fit_transform(df['skills'])
    
    # Combine features
    job_features = np.hstack([lsa_features, skills_matrix])
    
    return job_features, tfidf, svd, mlb, normalizer

job_features, tfidf, svd, mlb, normalizer = create_features(job_df)

## 4. Recommendation Models
class JobRecommender:
    def __init__(self, n_recommendations=5):
        self.n_recommendations = n_recommendations
        self.models = {}
        self.domain_indices = {}
        
    def train(self, df, features):
        """Train recommendation models per domain"""
        # First train a general model
        self.general_model = NearestNeighbors(n_neighbors=self.n_recommendations, metric='cosine')
        self.general_model.fit(features)
        
        # Then train domain-specific models
        for domain in df['domain'].unique():
            if pd.notna(domain) and str(domain).lower() != 'unknown':
                domain = str(domain).lower()
                domain_mask = df['domain'].str.lower() == domain
                self.domain_indices[domain] = domain_mask
                domain_features = features[domain_mask]
                
                if len(domain_features) > self.n_recommendations:
                    model = NearestNeighbors(
                        n_neighbors=min(self.n_recommendations, len(domain_features)), 
                        metric='cosine'
                    )
                    model.fit(domain_features)
                    self.models[domain] = model
    
    def recommend(self, query, features, df, domain_preference=None):
        """Recommend jobs based on text query"""
        # Preprocess query
        cleaned_query = clean_text(query)
        
        # Transform query to feature space
        query_tfidf = tfidf.transform([cleaned_query])
        query_lsa = normalizer.transform(svd.transform(query_tfidf))
        
        # Get skills from query
        query_skills = []
        for skill in mlb.classes_:
            if skill in cleaned_query:
                query_skills.append(skill)
        query_skills_vector = mlb.transform([query_skills])
        
        # Combine features
        query_features = np.hstack([query_lsa, query_skills_vector])
        
        # Get recommendations
        if domain_preference and str(domain_preference).lower() in self.models:
            # Domain-specific recommendations
            domain = str(domain_preference).lower()
            model = self.models[domain]
            domain_mask = self.domain_indices[domain]
            distances, indices = model.kneighbors(query_features)
            recommendations = df[domain_mask].iloc[indices[0]]
        else:
            # General recommendations
            distances, indices = self.general_model.kneighbors(query_features)
            recommendations = df.iloc[indices[0]]
        
        return recommendations

# Initialize and train the recommender
recommender = JobRecommender(n_recommendations=5)
recommender.train(job_df, job_features)

## 5. Example Usage
print("\nExample Recommendations:")
sample_query = "python developer with machine learning experience"
domain_pref = "IT"  # Optional: can be None for general recommendations

recommendations = recommender.recommend(sample_query, job_features, job_df, domain_pref)
print(recommendations[['title', 'company', 'domain', 'skills']].head())