from dotenv import load_dotenv

# Load environment variables FIRST, before importing modules that use them
load_dotenv()

from flask import Flask, render_template, request, jsonify
from main_agent import ice_break_with

# Set up the Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route("/")
def index():
    return render_template("index.html")

# Define the route for the process page
@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, picture_url = ice_break_with(name)
    
    # Check if the LinkedIn photo URL is accessible, use fallback if not
    fallback_image = "https://ui-avatars.com/api/?name=Eden+Marco&size=300&background=0D8ABC&color=fff"
    
    if picture_url:
        try:
            import requests
            response_check = requests.head(picture_url, timeout=5)
            if response_check.status_code != 200:
                picture_url = fallback_image
        except:
            picture_url = fallback_image
    else:
        picture_url = fallback_image
    
    # Structure the response to match what the frontend expects
    response = {
        "picture_url": picture_url,  # Frontend expects picture_url, not photoUrl
        "summary_and_facts": {       # Frontend expects this nested structure
            "summary": summary.summary,
            "facts": summary.facts
        },
        "ice_breakers": {            # Frontend expects this section
            "ice_breakers": [
                "What inspired you to work in AI/ML?",
                "Tell me about your experience at Google Cloud",
                "How do you balance technical work with teaching?"
            ]
        },
        "interests": {               # Frontend expects this section  
            "topics_of_interest": [
                "Large Language Models",
                "Backend Development",
                "AI/ML Education", 
                "Cloud Computing"
            ]
        }
    }
    
    return jsonify(response)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)