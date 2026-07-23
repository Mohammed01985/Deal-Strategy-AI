import os
import json
import re
from flask import Flask, render_template, request, jsonify
from groq import Groq

# ============================================================
# 🔐 GROQ SETUP
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it with: export GROQ_API_KEY='your_key'")

client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt, max_tokens=800, temperature=0.3):
    """Helper to call Groq and return clean text."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}")

# ============================================================
# 📄 ROUTES FOR TOOL PAGES (homepage is static, not served)
# ============================================================
app = Flask(__name__)

@app.route('/pitch.html')
def pitch_page():
    return render_template('pitch.html')

@app.route('/legaleagle.html')
def legaleagle_page():
    return render_template('legaleagle.html')

# ============================================================
# 📊 PITCH GENERATOR API
# ============================================================
@app.route('/generate_pitch', methods=['POST'])
def generate_pitch():
    data = request.json
    company_text = data.get('text', '').strip()
    style = data.get('style', 'professional')

    if not company_text:
        return jsonify({'error': 'Please provide company description.'}), 400

    try:
        # Step 1: Extract structured info
        extract_prompt = f"""
        You are a business analyst. Extract the following from the text below:
        - Industry (single word or short phrase)
        - Estimated annual revenue (range in USD, e.g., "$5M – $10M")
        - Two main pain points (short phrases)

        Return ONLY a valid JSON object with keys: industry, revenue, pain_points (list).

        Text: {company_text}
        """
        info_raw = call_groq(extract_prompt, max_tokens=150, temperature=0.3)
        if info_raw.startswith('```json'):
            info_raw = info_raw[7:-3]
        info = json.loads(info_raw)

        # Step 2: Generate pitch
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
        pitch = call_groq(pitch_prompt, max_tokens=200, temperature=0.7)

        return jsonify({
            'industry': info.get('industry', 'N/A'),
            'revenue': info.get('revenue', 'N/A'),
            'pain_points': info.get('pain_points', []),
            'pitch': pitch,
            'style': style
        })
    except Exception as e:
        return jsonify({'error': f'Pitch generation failed: {str(e)}'}), 500

# ============================================================
# ⚖️ LEGALEAGLE API
# ============================================================
@app.route('/analyze_contract', methods=['POST'])
def analyze_contract():
    data = request.json
    contract_text = data.get('text', '').strip()
    if not contract_text:
        return jsonify({'error': 'Please paste a contract or Terms of Service.'}), 400

    try:
        # Agent 1: Scorer
        score_prompt = f"""
        You are a contract fairness expert. Based on the following contract text, assign a Fairness Score from 0 to 100.
        - 100 = extremely fair to the user (all rights protected)
        - 0 = extremely unfair (one‑sided, predatory)
        Return ONLY the integer score, nothing else.

        Contract:
        {contract_text}
        """
        score_raw = call_groq(score_prompt, max_tokens=20, temperature=0.2)
        score_match = re.search(r'\b(\d{1,3})\b', score_raw)
        fairness_score = int(score_match.group(1)) if score_match else 50

        # Agent 2: Risks
        risk_prompt = f"""
        You are a legal risk analyst. Identify the 3 most dangerous hidden risks in the following contract.
        Return a JSON array of 3 strings, each describing a risk concisely (max 20 words).
        Example: ["Risk 1", "Risk 2", "Risk 3"]
        ONLY valid JSON, no extra text.

        Contract:
        {contract_text}
        """
        risk_raw = call_groq(risk_prompt, max_tokens=300, temperature=0.4)
        if risk_raw.startswith('```json'):
            risk_raw = risk_raw[7:-3]
        risks = json.loads(risk_raw)
        if not isinstance(risks, list) or len(risks) != 3:
            risks = ["Risk extraction failed"] * 3

        # Agent 3: Simplifier
        simple_prompt = f"""
        You are a plain‑language expert. Rewrite the following contract in plain English suitable for a 5th‑grade reading level.
        Keep all key points, but simplify legal jargon, long sentences, and passive voice.
        Return ONLY the rewritten text, no introductions.

        Contract:
        {contract_text}
        """
        plain_english = call_groq(simple_prompt, max_tokens=1200, temperature=0.5)

        # Agent 4: Negotiator
        negotiator_prompt = f"""
        You are a negotiation strategist. Based on the contract, suggest 3 specific counter‑offers or changes the user should request.
        Return a JSON array of 3 strings, each a concrete proposal.
        ONLY valid JSON, no extra text.

        Contract:
        {contract_text}
        """
        neg_raw = call_groq(negotiator_prompt, max_tokens=300, temperature=0.5)
        if neg_raw.startswith('```json'):
            neg_raw = neg_raw[7:-3]
        counter_offers = json.loads(neg_raw)
        if not isinstance(counter_offers, list) or len(counter_offers) != 3:
            counter_offers = ["Request clarification on liability", "Negotiate termination notice period", "Seek data deletion rights"]

        return jsonify({
            'fairness_score': fairness_score,
            'risks': risks,
            'plain_english': plain_english,
            'counter_offers': counter_offers
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# ============================================================
# 🚀 START THE SERVER
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
