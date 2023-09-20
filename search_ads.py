import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('youtube_sub_cleaned.csv')

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Subtitle'])

user_input = input("Enter your search query: ")


user_input_vector = tfidf_vectorizer.transform([user_input])


cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)

most_similar_index = cosine_similarities.argmax()

most_similar_target = df.at[most_similar_index, 'Title']

# Print the result
print(f"The most similar target value is: {most_similar_target}")
