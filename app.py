from flask import Flask, render_template , request, jsonify
from urllib.parse import urlparse
from main import analyze_url_readability

app = Flask(__name__)

@app.route('/')
def index():
    """
    Render the index page
    """
    return render_template('index.html')

@app.route('/analyze_readability', methods=['POST'])
def analyze_readability():   
    """
    Analyze the readability of a given URL
    """
    url = request.form['url']  

    # Validate the URL
    if not url or not urlparse(url).netloc:
        return jsonify({'error': 'Invalid URL'}), 400

    result = analyze_url_readability(url)

    if result:
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Failed to analyze the URL'}), 500

if __name__ == '__main__':
    """
    Run the application
    """
    app.run(debug=True)