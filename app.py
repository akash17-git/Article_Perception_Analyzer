from flask import Flask, request, jsonify
from main import analyze_article, fetch_article_from_url, is_url

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    article_text = data.get('text', '')
    
    # Check if the input is a URL and fetch the article
    if is_url(article_text):
        article_text = fetch_article_from_url(article_text)

    if not article_text:
        return jsonify({"error": "No valid article text found."}), 400
    
    # Analyze the article
    result = analyze_article(article_text)
    
    # Return the analysis result as a JSON response
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
