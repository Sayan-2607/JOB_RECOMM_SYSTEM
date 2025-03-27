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
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """Load and verify the data"""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully. First few rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def preprocess_data(df):
    """Preprocess the job data"""
    # Create a copy to avoid SettingWithCopyWarning
    processed_df = df.copy()
    
    # Clean text data - handle cases where columns might not exist
    text_columns = ['Title', 'title', 'Job Title', 'job_title', 
                   'Description', 'description', 'Job Description', 'job_description']
    
    # Find which text columns exist in the dataframe
    existing_cols = [col for col in text_columns if col in processed_df.columns]
    
    if len(existing_cols) >= 2:
        # Use the first available column for title and description
        title_col = existing_cols[0]
        desc_col = existing_cols[1]
        
        processed_df['cleaned_title'] = processed_df[title_col].apply(clean_text)
        processed_df['cleaned_desc'] = processed_df[desc_col].apply(clean_text)
    else:
        # Fallback if columns not found
        processed_df['cleaned_title'] = ""
        processed_df['cleaned_desc'] = ""
    
    # Extract skills (simplified approach)
    common_skills = {
        'it': ['python', 'java', 'sql', 'machine learning', 'aws', 'cloud', 'developer', 'software'],
        'healthcare': ['patient', 'medical', 'health', 'care', 'nursing', 'hospital', 'doctor'],
        'finance': ['financial', 'accounting', 'excel', 'analysis', 'banking', 'investment'],
        'marketing': ['marketing', 'social media', 'seo', 'content', 'digital', 'advertising']
    }
    
    def extract_skills(row):
        domain = str(row.get('domain', '') or row.get('industry', '') or row.get('category', '')).lower()
        desc = row.get('cleaned_desc', '')
        skills = []
        
        # Find matching domain
        for dom, skill_list in common_skills.items():
            if dom in domain:
                skills.extend([skill for skill in skill_list if skill in desc])
        
        # Add general skills if no domain specific ones found
        if not skills:
            general_skills = ['communication', 'team', 'management', 'leadership', 'problem solving']
            skills.extend([skill for skill in general_skills if skill in desc])
        
        return list(set(skills))
    
    processed_df['skills'] = processed_df.apply(extract_skills, axis=1)
    
    return processed_df

def create_features(df):
    """Create features for job recommendations"""
    try:
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
    except Exception as e:
        print(f"Error in feature creation: {str(e)}")
        return None, None, None, None, None

def main():
    # Load data
    file_path = "Combined_Jobs_Final.csv"
    job_df = load_data(file_path)
    
    if job_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    job_df = preprocess_data(job_df)
    
    # Create features
    job_features, tfidf, svd, mlb, normalizer = create_features(job_df)
    
    if job_features is None:
        print("Failed to create features. Exiting.")
        return
    
    # Recommendation system
    class JobRecommender:
        def __init__(self, n_recommendations=5):
            self.n_recommendations = n_recommendations
            self.models = {}
            self.domain_indices = {}
            
        def train(self, df, features):
            """Train recommendation models"""
            # General model
            self.general_model = NearestNeighbors(n_neighbors=self.n_recommendations, metric='cosine')
            self.general_model.fit(features)
            
            # Domain-specific models
            domain_col = next((col for col in ['domain', 'industry', 'category'] if col in df.columns), None)
            
            if domain_col:
                for domain in df[domain_col].unique():
                    if pd.notna(domain):
                        domain = str(domain).lower()
                        domain_mask = df[domain_col].str.lower() == domain
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
            """Recommend jobs based on query"""
            try:
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
                    domain = str(domain_preference).lower()
                    model = self.models[domain]
                    domain_mask = self.domain_indices[domain]
                    distances, indices = model.kneighbors(query_features)
                    recommendations = df[domain_mask].iloc[indices[0]]
                else:
                    distances, indices = self.general_model.kneighbors(query_features)
                    recommendations = df.iloc[indices[0]]
                
                return recommendations
            except Exception as e:
                print(f"Error in recommendation: {str(e)}")
                return pd.DataFrame()
    
    # Initialize and train the recommender
    recommender = JobRecommender(n_recommendations=5)
    recommender.train(job_df, job_features)
    
    # Example usage
    print("\nExample Recommendations:")
    sample_query = "python developer with machine learning experience"
    domain_pref = "IT"  # Optional: can be None for general recommendations
    
    recommendations = recommender.recommend(sample_query, job_features, job_df, domain_pref)
    
    if not recommendations.empty:
        print(recommendations[['cleaned_title', 'skills']].head())
    else:
        print("No recommendations could be generated.")

if __name__ == "__main__":
    main()