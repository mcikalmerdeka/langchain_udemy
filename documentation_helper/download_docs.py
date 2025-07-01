import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
from collections import deque
import time

def scrape_docs():
    """
    Recursively scrapes the LangChain API documentation.
    """
    start_url = "https://python.langchain.com/api_reference/"
    output_dir = "documentation_helper/langchain-docs-newest/"

    # First, find the actual base URL after any redirects
    try:
        initial_response = requests.get(start_url, timeout=10)
        initial_response.raise_for_status()
        base_url = initial_response.url
        print(f"Starting crawl from redirected URL: {base_url}")
    except requests.RequestException as e:
        print(f"Failed to get initial URL: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    urls_to_visit = deque([base_url])
    visited_urls = set()
    session = requests.Session()

    while urls_to_visit:
        current_url = urls_to_visit.popleft()

        # Normalize URL by removing fragment
        current_url = urllib.parse.urljoin(current_url, urllib.parse.urlparse(current_url).path)

        if current_url in visited_urls:
            continue

        print(f"Scraping: {current_url}")
        visited_urls.add(current_url)

        try:
            response = session.get(current_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {current_url}: {e}")
            continue

        # Create a file path that mirrors the URL structure
        relative_path = current_url.replace(base_url, '')
        if not relative_path:
            relative_path = 'index.html'
        elif relative_path.endswith('/'):
            relative_path += 'index.html'

        file_path = os.path.join(output_dir, relative_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links on the page and add them to the queue
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urllib.parse.urljoin(current_url, href)

            # Only visit pages within the API documentation
            if full_url.startswith(base_url):
                # Normalize URL before adding to queue
                full_url_no_fragment = urllib.parse.urljoin(full_url, urllib.parse.urlparse(full_url).path)
                if full_url_no_fragment not in visited_urls:
                    urls_to_visit.append(full_url_no_fragment)
        
        # Be polite to the server
        time.sleep(0.1)

    print("Scraping complete.")

if __name__ == "__main__":
    scrape_docs()

