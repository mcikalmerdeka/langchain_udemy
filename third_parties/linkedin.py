import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Create a function to scrape a linkedin profile
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """Scrape information from Linkedin profiles,
    Manually scrape the information from the LinkedIn profile
    """

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/mcikalmerdeka/89c47d38232381cdbcbb552851be6425/raw/6150c766bd5a1b7d32657fd8ebc3a11c0c630404/cikal-merdeka-scrapin.json"

        # # If using the course example, use this url
        # linkedin_profile_url = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"
        
        response = requests.get(
            url=linkedin_profile_url,
            timeout=10
        )
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.getenv("SCRAPIN_API_KEY"),
            "linkedInUrl": linkedin_profile_url
        }

        response = requests.get(
            url=api_endpoint,
            params=params,
            timeout=10
        )
    
    # Get the data from the response
    data = response.json().get("person")
    
    # Filter only the necessary data (output will be a dictionary)
    filtered_data = {
        k: v
        for k, v in data.items()
        if v not in ["", "", [], None]
        and k not in ["certifications"]
    }

    return filtered_data

if __name__ == "__main__":
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/mcikalmerdeka/", mock=True))