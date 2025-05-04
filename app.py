import pandas as pd
import numpy as np
from pymongo import MongoClient
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import json
import re

# Use secrets from Streamlit
MONGODB_URI = st.secrets["MONGODB_URI"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

class AssessmentSearchSystem:
    def __init__(self, mongodb_uri, gemini_api_key=None, collection_name="tests"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['assessment_search']
        self.collection = self.db[collection_name]
        
        genai.configure(api_key=gemini_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    
    def create_document_text(self, row):
        document_text = f"""
        Name: {row.get('Name', '')}
        Description: {row.get('Description', '')}
        Test Type: {row.get('Test Type', '')} 
        Job Levels: {row.get('Job Levels', '')}
        Languages: {row.get('Languages', '')}
        Assessment Length: {row.get('Assessment Length', '')}
        """
        return document_text.strip()
    
    def generate_embedding(self, text):
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def refine_query(self, query):
        prompt = f"""
        the user query is {query}
        You are a search query optimizer for an testing solutions database. 
        The database contains records with these fields:
        - Name: Testing solution names
        - Description: Detailed description of the test
        - Test Type: Categories like 'Ability & Aptitude','Biodata & Situational Judgement','Competencies','Development & 360','Assessment Exercises','Knowledge & Skills','Personality & Behavior','Simulations'
        - Job Levels: Target job levels (Entry-level, Mid-Professional, Manager, etc.)
        - Languages: Available languages
        - Assessment Length: Duration of the test ,if nothing is mentioned dont use this field. I query like about an hour then assessment length should be <=60
        If query like 30-40 mins then assessment length should be 30-40
        
        Always include description of the assessment in the refined query.
        The spoken languages in the query should be kept in the Languages field.
        The test types if many should be separated by commas.
        Given the following user query, extract key search terms and criteria relevant to finding 
        matching assessments. Format your response as a clean, refined search query that highlights 
        the most important requirements.Do not output any explanation and additional text, only give the refined query.

        Example - 

        Name: .NET MVC (New)
        Description: knowledge of Model-View-Controller (MVC) architecture, validation and .NET framework.
        Test Type: Knowledge & Skills 
        Job Levels: Mid-Professional, Professional Individual Contributor,
        Languages: English (USA),
        Assessment Length: 17
        """
        
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()
    
    def process_csv_and_create_embeddings(self, csv_file):
        df = pd.read_csv(csv_file)
        print(f"Processing {len(df)} records from {csv_file}")
        for index, row in df.iterrows():
            document_text = self.create_document_text(row)
            embedding = self.generate_embedding(document_text)
            if embedding:
                document = {
                    'name': row.get('Name', ''),
                    'url': row.get('URL', ''),
                    'remote_testing': row.get('Remote Testing', ''),
                    'adaptive': row.get('Adaptive/IRT', ''),
                    'test_type': row.get('Test Type', ''),
                    'description': row.get('Description', ''),
                    'job_levels': row.get('Job Levels', ''),
                    'languages': row.get('Languages', ''),
                    'assessment_length': row.get('Assessment Length', ''),
                    'text': document_text,
                    'embedding': embedding
                }
                self.collection.update_one(
                    {'name': document['name']},
                    {'$set': document},
                    upsert=True
                )
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1} records...")
        print(f"Completed processing {len(df)} records.")

    def search(self, query, limit):
        refined = self.refine_query(query)
        print(refined)
        query_embedding = self.generate_embedding(refined)
        if not query_embedding:
            return []

        length_pattern = r'Assessment Length:\s*(<=|>=|)(\d+)(?:-(\d+)|)'
        length_match = re.search(length_pattern, refined)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 600,
                    "limit": 100
                }
            }
        ]

        if length_match:
            operator = length_match.group(1)
            first_num = float(length_match.group(2))
            second_num = length_match.group(3)

            if operator == "<=":
                length_filter = {"$lte": first_num}
            elif operator == ">=":
                length_filter = {"$gte": first_num}
            elif second_num:
                length_filter = {"$gte": first_num, "$lte": float(second_num)}
            else:
                length_filter = {"$eq": first_num}

            pipeline.append({"$match": {"assessment_length": length_filter}})

        pipeline.append({
            "$project": {
                "_id": 0,
                "name": 1,
                "url": 1,
                "remote_testing": 1,
                "adaptive": 1,
                "test_type": 1,
                "description": 1,
                "job_levels": 1,
                "languages": 1,
                "assessment_length": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        })

        pipeline.append({"$limit": limit})

        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

def streamlit_ui():
    st.title("🧠 SHL Assessment Search Tool")

    search_system = AssessmentSearchSystem(MONGODB_URI, GEMINI_API_KEY)

    query = st.text_area("🔍 Enter your assessment query:", height=200)

    if st.button("Search"):
        if query.strip():
            with st.spinner("Searching..."):
                results = search_system.search(query, 10)
            if results:
                st.success("Here are the top matching assessments:")
                for i, result in enumerate(results):
                    st.markdown(f"### {i+1}. {result['name']}")
                    st.markdown(f"[🔗 Assessment Link]({result['url']})")
                    st.write(f"**Test Type**: {result['test_type']}")
                    st.write(f"**Duration**: {result['assessment_length']} mins")
                    st.write(f"**Remote Testing**: {result['remote_testing']}")
                    st.write(f"**Adaptive**: {result['adaptive']}")
                    st.write(f"**Languages**: {result['languages']}")
                    st.write(f"**Job Levels**: {result['job_levels']}")
                    st.markdown("---")
            else:
                st.warning("No matching assessments found.")
        else:
            st.warning("Please enter a query to search.")

if __name__ == "__main__":
    streamlit_ui()
