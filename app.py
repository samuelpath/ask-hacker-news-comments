import streamlit as st
import requests
import cohere
import json
from typing import Any

# Cohere Client
co = cohere.Client(st.secrets["COHERE_API_KEY"])

# Hackernews Client
class HackernewsClient:
    BASE_URL = "https://hn.algolia.com/api/v1"

    def __init__(self, search_limit=7):
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
    stripped_result = {key: str(result.get(key, '')) for key in ['author', 'comment_text', 'story_title']}

    return stripped_result

# Streamlit App
st.title('Ask Hacker News Comments')

st.write("This mini-app uses Cohere's chat API to answer questions based on the context of Hacker News comments.")
st.write("It uses the Hacker News API to search for comments based on the user's question.")
st.write("Since Cohere's command model is limited to 4096 tokens, the context information is limited to 7 search results.")

user_input = st.text_input("Your question: ")

if st.button("Submit"):
    # Hackernews search results
    search_results = search(user_input)
    
    stripped_search_results = [decorate_and_serialize_search_result(result) for result in search_results]
    stripped_search_results_text = json.dumps(stripped_search_results, indent=4)
    
    prompt = f"""
      Context information is below.
      ---------------------
      {stripped_search_results_text}
      ---------------------
      Given the context information and not prior knowledge, answer the query below.
      Query: {user_input}
      Answer: 
    """

    # Cohere chat response
    response = co.chat(
        prompt, 
        model="command", 
        temperature=0.5
    )

    # Clear loading message
    st.empty()

    st.write(response.text)

    # Display raw search results
    st.write("Raw Search Results: ")
    for result in search_results:
        st.json(result)
