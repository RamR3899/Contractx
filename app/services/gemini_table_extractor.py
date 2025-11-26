"""
Gemini Table Extraction with Merged Cell Handling
"""

import os
import json
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image
import requests
from app.config.config import Config

class GeminiTableExtractor:
    """Extract complete table structure using Gemini with merged cell handling"""
    
    def __init__(self):
        self.API_KEY = Config.GEMINI_API_KEY
        self.MODEL = "gemini-2.0-flash"
        self.ENDPOINT = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.MODEL}:generateContent?key={self.API_KEY}"
        )
        
        # Rate limiting
        self.request_delay = float(os.getenv("GEMINI_REQUEST_DELAY", "3.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "60"))
        
        # Output directory for table images
        self.output_dir = "extracted_tables"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[OK] GeminiTableExtractor initialized")
        print(f"  - Model: {self.MODEL}")
        print(f"  - Output: {self.output_dir}/")
    
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status"""
        return {
            "model": self.MODEL,
            "api_key_present": bool(self.API_KEY),
            "output_dir": self.output_dir
        }
    
    async def extract_table(
        self,
        page_image: Image.Image,
        bbox: List[float],
        confidence: float,
        page_number: int,
        table_index: int
    ) -> Dict[str, Any]:
        """
        Extract complete table structure from detected region
        
        Returns complete table JSON with merged cell handling
        """
        # Crop table with adaptive padding
        table_image = self._crop_table(page_image, bbox)
        
        # Save table image
        image_filename = f"page_{page_number}_table_{table_index}.png"
        image_path = os.path.join(self.output_dir, image_filename)
        table_image.save(image_path)
        
        # Extract with retry
        table_data = await self._extract_with_retry(
            table_image=table_image,
            page_number=page_number,
            table_index=table_index
        )
        
        # Add metadata
        table_data["page_number"] = page_number
        table_data["bbox"] = bbox
        table_data["confidence"] = confidence
        table_data["image_file"] = image_path
        table_data["source"] = self.MODEL
        
        return table_data
    
    def _crop_table(self, page_image: Image.Image, bbox: List[float]) -> Image.Image:
        """Crop table with adaptive padding"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Adaptive padding based on table size
        w = x2 - x1
        h = y2 - y1
        
        pad_x = max(40, min(int(w * 0.10), 100))
        pad_y = max(50, min(int(h * 0.12), 120))
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(page_image.width, x2 + pad_x)
        y2 = min(page_image.height, y2 + pad_y)
        
        return page_image.crop((x1, y1, x2, y2))
    
    async def _extract_with_retry(
        self,
        table_image: Image.Image,
        page_number: int,
        table_index: int
    ) -> Dict[str, Any]:
        """Extract table with automatic retry"""
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Rate limiting
                await asyncio.sleep(self.request_delay)
                
                # Call Gemini
                result = self._call_gemini_api(table_image)
                
                if result:
                    print(f"      [OK] Table extracted: {result.get('total_rows', 0)}x{result.get('total_columns', 0)}")
                    return result
                else:
                    raise Exception("Gemini returned empty result")
                    
            except Exception as e:
                error_msg = str(e).lower()
                print(f"      [!] Extraction error (attempt {attempt}): {e}")
                
                # Check for quota error
                if any(kw in error_msg for kw in ['quota', 'rate limit', '429', 'resource exhausted']):
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * 3 * (2 ** (attempt - 1))
                        print(f"      [i] Quota error - waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        return self._empty_table_result(page_number, table_index)
                else:
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(wait_time)
                    else:
                        return self._empty_table_result(page_number, table_index)
        
        return self._empty_table_result(page_number, table_index)
    
    def _call_gemini_api(self, table_image: Image.Image) -> Optional[Dict[str, Any]]:
        """Call Gemini API to extract table structure"""
        
        # Convert image to base64
        buffer = BytesIO()
        table_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create prompt
        prompt = self._create_extraction_prompt()
        
        # API payload
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.0,
                "candidateCount": 1
            }
        }
        
        # Make request
        response = requests.post(
            self.ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text[:500]}")
        
        # Parse response
        resp_json = response.json()
        text_output = self._extract_text_from_response(resp_json)
        
        if not text_output:
            raise Exception("No text output from Gemini")
        
        # Parse JSON
        return self._parse_json_output(text_output)
    
    def _create_extraction_prompt(self) -> str:
        """Create comprehensive extraction prompt"""
        return """Extract COMPLETE table structure from this image.

CRITICAL REQUIREMENTS:
1. Extract ALL rows (no truncation!)
2. Extract ALL columns
3. Handle merged cells correctly
4. Preserve exact cell values

MERGED CELLS HANDLING:
- If cell spans multiple rows → REPEAT value in each row
- If cell spans multiple columns → REPEAT value in each column
- Mark merged regions in "merged_cells" field

EXAMPLE:
Visual table with merged cell spanning rows 1-2:
┌──────────┬──────────┐
│ Item 1   │ Status   │
├──────────┤          │
│ Item 2   │          │
└──────────┴──────────┘

Output:
{
  "headers": ["Item", "Status"],
  "rows": [
    ["Item 1", "Status"],
    ["Item 2", "Status"]
  ],
  "has_merged_cells": true,
  "merged_cells": "Column 2 'Status' spans rows 1-2"
}

RETURN ONLY THIS JSON:
{
  "table_id": "T<idx>",
  "table_title": "title or empty string",
  "position": "top/center/bottom",
  "size": "small/medium/large/full-width",
  "table_type": "financial/schedule/comparison/data/specification",
  "headers": ["Column1", "Column2", ...],
  "rows": [
    ["row1col1", "row1col2", ...],
    ["row2col1", "row2col2", ...]
  ],
  "total_rows": <number>,
  "total_columns": <number>,
  "has_merged_cells": true/false,
  "merged_cells": "description or null",
  "data_types": ["text", "number", "date"],
  "notes": "footnotes or empty string"
}

NO markdown, NO explanations, ONLY JSON!"""
    
    def _extract_text_from_response(self, resp_json: Dict[str, Any]) -> Optional[str]:
        """Extract text from Gemini response"""
        try:
            candidate = resp_json.get("candidates", [None])[0]
            if not candidate:
                return None
            
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            texts = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
            
            return "\n".join(texts).strip() if texts else None
            
        except Exception:
            return None
    
    def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from text output"""
        # Strip code fences
        text = text.strip()
        if text.startswith("```"):
            first = text.find("```")
            last = text.rfind("```")
            if last > first:
                text = text[first + 3:last].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
        
        # Try to parse
        try:
            data = json.loads(text)
            
            # Validate required fields
            required = ["headers", "rows", "total_rows", "total_columns", "has_merged_cells"]
            if all(field in data for field in required):
                return data
            else:
                print(f"      [!] Missing required fields in response")
                return None
                
        except json.JSONDecodeError as e:
            print(f"      [!] JSON parse error: {e}")
            
            # Try to find JSON in text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except:
                    pass
            
            return None
    
    def _empty_table_result(self, page_number: int, table_index: int) -> Dict[str, Any]:
        """Return empty table structure on failure"""
        return {
            "table_id": f"T{table_index}",
            "table_title": "",
            "position": "unknown",
            "size": "unknown",
            "table_type": "unknown",
            "headers": [],
            "rows": [],
            "total_rows": 0,
            "total_columns": 0,
            "has_merged_cells": False,
            "merged_cells": None,
            "data_types": [],
            "notes": "",
            "error": "Extraction failed after retries"
        }