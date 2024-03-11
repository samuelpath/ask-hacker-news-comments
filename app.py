import streamlit as st
import requests
import cohere
from bs4 import BeautifulSoup
from typing import Any

# Cohere Client
co = cohere.Client(st.secrets["COHERE_API_KEY"])

# Hackernews Client
class HackernewsClient:
    BASE_URL = "https://hn.algolia.com/api/v1"

    def __init__(self, search_limit=25):
        self.search_limit = search_limit

    def search(self, query):
        url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "hitsPerPage": self.search_limit,
            "tags": "comment"
        }
        response = requests.get(
            url,
            params=params,
        )

        if response.status_code != 200:
            raise Exception(
                f"Error searching HackerNews with query: `{query}`."
            )

        return response.json()["hits"]

    def get_item(self, item_id):
        url = f"{self.BASE_URL}/items/{item_id}"
        response = requests.get(
            url,
        )

        if response.status_code != 200:
            return {}

        return response.json()

hn_client = HackernewsClient()

# Search Provider
def search(query) -> list[dict[str, Any]]:
    search_results = hn_client.search(query)

    results = []
    for page in search_results:
        results.append(decorate_and_serialize_search_result(page))

    return results

def decorate_and_serialize_search_result(result):
    # Only keep the fields of interest
    stripped_result = {key: str(result.get(key, '')) for key in ['author', 'created_at', 'comment_text', 'story_title']}

    return stripped_result

# Streamlit App
st.title('Ask Hacker News Comments')

st.write("This mini-app uses Cohere's chat API to answer questions based on the context of Hacker News comments.")
st.write("It uses the Hacker News API to search for comments based on the user's question.")
st.write("The search results are then reranked using Cohere's rerank model.")
st.write("At first we used the command model limited to 4096 tokens and thus limited the context information to 7 search results")
st.write("Now with access to the command-r's 128k context window, we retrieve 25 results and rerank them to get the top 12 results.")
st.write("We also display the raw search results and the prompt used for the chat API below the model's answer.")

user_input = st.text_input("Your question: ")

if st.button("Submit"):
    # Hackernews search results
    search_results = search(user_input)
    
    # we only keep the comment_text's raw text for the rerank model
    comment_texts = [BeautifulSoup(search_result['comment_text'], features="html.parser").get_text() for search_result in search_results]
    
    # we rerank the search results using the rerank model, keeping the top 12 results out of the initial 25
    reranked_results = co.rerank(query=user_input, documents=comment_texts, top_n=12, model='rerank-english-v2.0')
    
    # we only keep the text of the reranked results for the prompt    
    reranked_results_text_only = [result.document['text'] for result in reranked_results]
    
    # context information formatted for the prompt
    reranked_results_formatted = '\n\n - '.join(reranked_results_text_only)
    
    prompt = f"""
      Context information is below.
      
      {reranked_results_formatted}
      
      Given the context information and not prior knowledge, answer the query below.
      
      Query: {user_input}
      
      Answer: 
    """

    # Cohere chat response
    response = co.chat(
        prompt, 
        model="command-r", 
        temperature=0.5
    )

    # Clear loading message
    st.empty()

    st.write("Answer: ")
    st.write(response.text)

    # Display raw search results
    st.write("Raw Search Results: ")
    for result in search_results:
        st.json(result)
        
    # Display the prompt used for the chat API
    st.write("Prompt: ")
    st.write(prompt)
    
