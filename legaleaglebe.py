import os
import json
import re
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# Load API key from environment (DO NOT hardcode!)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable")

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

@app.route('/')
def index():
    return render_template('contract.html')

@app.route('/analyze_contract', methods=['POST'])
def analyze_contract():
    data = request.json
    contract_text = data.get('text', '').strip()
    if not contract_text:
        return jsonify({'error': 'Please paste a contract or Terms of Service.'}), 400

    try:
        # ---- Agent 1: The Scorer (Fairness Score 0-100) ----
        score_prompt = f"""
You are a contract fairness expert. Based on the following contract text, assign a Fairness Score from 0 to 100.
- 100 = extremely fair to the user (all rights protected)
- 0 = extremely unfair (one‑sided, predatory)
Return ONLY the integer score, nothing else.

Contract:
{contract_text}
"""
        score_raw = call_groq(score_prompt, max_tokens=20, temperature=0.2)
        # Extract first number found
        score_match = re.search(r'\b(\d{1,3})\b', score_raw)
        fairness_score = int(score_match.group(1)) if score_match else 50

        # ---- Agent 2: Risk Finder (3 hidden dangers) ----
        risk_prompt = f"""
You are a legal risk analyst. Identify the 3 most dangerous hidden risks in the following contract.
Return a JSON array of 3 strings, each describing a risk concisely (max 20 words).
Example: ["Risk 1 description", "Risk 2 description", "Risk 3 description"]
ONLY valid JSON, no extra text.

Contract:
{contract_text}
"""
        risk_raw = call_groq(risk_prompt, max_tokens=300, temperature=0.4)
        # Clean markdown if present
        if risk_raw.startswith('```json'):
            risk_raw = risk_raw[7:-3]
        risks = json.loads(risk_raw)
        if not isinstance(risks, list) or len(risks) != 3:
            risks = ["Risk extraction failed"] * 3

        # ---- Agent 3: The Simplifier (Plain English) ----
        simple_prompt = f"""
You are a plain‑language expert. Rewrite the following contract in plain English suitable for a 5th‑grade reading level.
Keep all key points, but simplify legal jargon, long sentences, and passive voice.
Return ONLY the rewritten text, no introductions.

Contract:
{contract_text}
"""
        plain_english = call_groq(simple_prompt, max_tokens=1200, temperature=0.5)

        # ---- Agent 4: The Negotiator (3 counter‑offers) ----
        negotiator_prompt = f"""
You are a negotiation strategist. Based on the contract, suggest 3 specific counter‑offers or changes the user should request.
Return a JSON array of 3 strings, each a concrete proposal (e.g., "Remove the automatic renewal clause").
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

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({
        'fairness_score': fairness_score,
        'risks': risks,
        'plain_english': plain_english,
        'counter_offers': counter_offers
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)   # use a different port than pitch generator
