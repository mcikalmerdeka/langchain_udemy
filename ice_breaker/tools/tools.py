# """This file is going to contain all of the tools that we are going to use in the project"""

from langchain_tavily import TavilySearch

def get_profile_url_tavily(query: str) -> str:
    """Searches for Linkedin or Twitter profile page"""

    search = TavilySearch(max_results=2)
    results = search.run(query)

    return results