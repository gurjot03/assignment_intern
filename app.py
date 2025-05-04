import streamlit as st
from search_system.assessment_search import AssessmentSearchSystem
import os
from dotenv import load_dotenv

load_dotenv()

def main():

    st.title("Testing Solutions Search App")
    
    mongodb_uri = st.secrets['MONGODB_URI']
    
    search_system = AssessmentSearchSystem(mongodb_uri)

    query = st.text_input("Enter your query:")
    
    if st.button("Search"):
        if query:
            results = search_system.search(query, 10)
            if results:
                st.subheader("Search Results:")
                for i, result in enumerate(results):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**#{i + 1}**")
                    with col2:
                        st.markdown(f"**{result['name']}**")
                        st.markdown(f"🔗 [{result['url']}]({result['url']})")
                        
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            st.write(f"⏱️ Duration: {result['assessment_length']}")
                            st.write(f"🔄 Remote Testing: {result['remote_testing']}")
                        with meta_col2:
                            st.write(f"📝 Test Type: {result['test_type']}")
                            st.write(f"🎯 Adaptive: {result['adaptive']}")
                        
                        st.markdown("---")
            else:
                st.write("No results found.")
        else:
            st.warning("Please enter a query to search.")

if __name__ == "__main__":
    main()