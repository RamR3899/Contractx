import fitz  # PyMuPDF
from typing import List, Dict, Any
from PIL import Image
import io
import base64

class PDFProcessor:
    """PDF Processor for ContractX – extracts text, images, PIL page image, and raw bytes"""

    async def extract_pages(self, pdf_path: str, dpi: int = 350) -> List[Dict[str, Any]]:
        """
        Extract each page as:
         - text
         - inline images
         - PIL image (for table detection)
         - image_bytes (PNG raw bytes for Gemini)
        """

        pages = []

        try:
            doc = fitz.open(pdf_path)
            print(f"[OK] PDF opened: {len(doc)} pages")

            for page_num in range(len(doc)):
                page = doc[page_num]

                # -----------------------------
                # 1️⃣ Extract TEXT
                # -----------------------------
                text = page.get_text("text")

                # -----------------------------
                # 2️⃣ Extract INLINE IMAGES
                # -----------------------------
                images = []
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                        images.append({
                            "image_id": f"page_{page_num + 1}_img_{img_index + 1}",
                            "format": base_image["ext"],
                            "data": img_b64
                        })

                    except Exception as e:
                        print(f"[!] Error extracting image {img_index} on page {page_num + 1}: {e}")

                # -----------------------------
                # 3️⃣ Render FULL PAGE as PIL Image (for table detection)
                # -----------------------------
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Also store image_bytes for Gemini API
                image_bytes = img_bytes

                # -----------------------------
                # 4️⃣ Build Page Object
                # -----------------------------
                page_data = {
                    "page_number": page_num + 1,
                    "text": text,
                    "text_length": len(text),
                    "images": images,
                    "image_count": len(images),

                    # Required for table detection + extraction
                    "pil_image": pil_image,
                    "image_bytes": image_bytes,

                    "dimensions": {
                        "width": page.rect.width,
                        "height": page.rect.height
                    },

                    # Raw PyMuPDF page object if needed later
                    "raw_page": page
                }

                pages.append(page_data)

            doc.close()
            return pages

        except Exception as e:
            raise Exception(f"Error extracting PDF pages: {str(e)}")
