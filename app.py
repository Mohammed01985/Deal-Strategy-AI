from flask import Flask, render_template, request, jsonify
from groq import Groq
import json

app = Flask(__name__)

# 🔑 PASTE YOUR NEW GROQ API KEY HERE
GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

@app.route('/')
def index():
    return render_template('pitch.html')

@app.route('/generate_pitch', methods=['POST'])
def generate_pitch():
    data = request.json
    company_text = data.get('text', '').strip()
    style = data.get('style', 'professional')

    if not company_text:
        return jsonify({'error': 'Please provide company description.'}), 400

    # Step 1: Extract structured info
    extract_prompt = f"""
    You are a business analyst. Extract the following from the text below:
    - Industry (single word or short phrase)
    - Estimated annual revenue (range in USD, e.g., "$5M – $10M")
    - Two main pain points (short phrases)

    Return ONLY a valid JSON object with keys: industry, revenue, pain_points (list).

    Text: {company_text}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.3,
            max_tokens=150
        )
        info_raw = response.choices[0].message.content.strip()
        if info_raw.startswith('```json'):
            info_raw = info_raw[7:-3]
        info = json.loads(info_raw)
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

    # Step 2: Generate pitch based on style
    style_map = {
        'professional': 'a professional, persuasive 3-sentence pitch for a B2B sales meeting.',
        'investor': 'a compelling 3-sentence pitch for venture capitalists, highlighting market opportunity and scalability.',
        'elevator': 'a very short, punchy 30-second elevator pitch (2-3 sentences).',
        'casual': 'a friendly, conversational 3-sentence pitch suitable for networking events.'
    }
    pitch_prompt = f"""
    Write {style_map.get(style, 'a professional 3-sentence pitch')} for a company based on this description:

    {company_text}

    The pitch should be clear, convincing, and tailored to the style.
    Return ONLY the pitch text, no extra commentary.
    """
    try:
        response2 = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": pitch_prompt}],
            temperature=0.7,
            max_tokens=200
        )
        pitch = response2.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({'error': f'Pitch generation failed: {str(e)}'}), 500

    return jsonify({
        'industry': info.get('industry', 'N/A'),
        'revenue': info.get('revenue', 'N/A'),
        'pain_points': info.get('pain_points', []),
        'pitch': pitch,
        'style': style
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
