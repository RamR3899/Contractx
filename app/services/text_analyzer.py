"""
Text Analysis using Gemini
Extracts sections, clauses, entities (TEXT ONLY - no tables)
"""

import google.generativeai as genai
import os
import json
import asyncio
from typing import Dict, Any
from app.config.config import Config

class TextAnalyzer:
    """Analyze document text for structure and entities"""
    
    def __init__(self):
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.request_delay = float(os.getenv("GEMINI_REQUEST_DELAY", "3.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "60"))
        
        print("[OK] TextAnalyzer initialized")
    
    async def analyze_text(self, page_number: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze text content (no tables)
        
        Returns structure with sections, entities
        """
        text = page_data['text']
        
        for attempt in range(1, self.max_retries + 1):
            try:
                await asyncio.sleep(self.request_delay)
                
                prompt = self._create_prompt(page_number, text)
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean JSON
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                data = json.loads(result_text.strip())
                data['sections_count'] = len(data.get('sections', []))
                
                return data
                
            except json.JSONDecodeError as e:
                print(f"    [!] JSON error (attempt {attempt})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                else:
                    return self._empty_result()
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(kw in error_msg for kw in ['quota', 'rate limit', '429', 'resource exhausted']):
                    print(f"    [!] Quota error (attempt {attempt})")
                    if attempt < self.max_retries:
                        wait = self.retry_delay * 3 * (2 ** (attempt - 1))
                        print(f"    [i] Waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        return self._empty_result()
                else:
                    print(f"    [!] Error: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    else:
                        return self._empty_result()
        
        return self._empty_result()
    
    def _create_prompt(self, page_number: int, text: str) -> str:
        """Create analysis prompt"""
        return f"""Analyze this page from a contract/legal document.

IMPORTANT: Extract content from PLAIN TEXT ONLY. DO NOT extract from tables!

PAGE {page_number}:
{text[:5000]}

Extract:
1. Sections/Headings/Clauses (from prose only, NOT from tables)
2. Entities:
   - buyer_name
   - seller_name  
   - dates
   - deadlines
   - alerts/critical items

Return ONLY JSON:
{{
  "sections": [
    {{
      "heading": "Section title",
      "heading_id": "1",
      "clauses": [
        {{
          "clause": "Clause text", 
          "clause_id": "1.1",
          "sub_clauses": [
            {{
              "sub_clause": "Sub-clause text",
              "sub_clause_id": "1.1.1"
            }}
          ]
        }}
      ]
    }}
  ],
  "entities": {{
    "buyer_name": "name or null",
    "seller_name": "name or null",
    "dates": ["date1", "date2"],
    "deadlines": ["deadline1"],
    "alerts": ["critical item 1"]
  }}
}}

NO markdown, NO explanations, ONLY JSON!"""
    
    def _empty_result(self) -> Dict[str, Any]:
        """Empty result on failure"""
        return {
            "sections": [],
            "sections_count": 0,
            "entities": {
                "buyer_name": None,
                "seller_name": None,
                "dates": [],
                "deadlines": [],
                "alerts": []
            },
            "error": "Analysis failed"
        }