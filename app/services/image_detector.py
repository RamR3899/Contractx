"""
Advanced Image/Visual Detection using OpenCV
Detects charts, graphs, diagrams, Gantt charts, and other visuals
Based on edge detection, entropy, and morphological operations
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from PIL import Image

class ImageDetector:
    """Detect non-table visuals (charts, graphs, diagrams, images)"""
    
    def __init__(self):
        # Strict detection parameters (default)
        self.DPI = 250
        self.MIN_AREA = 20000
        self.MIN_W = 120
        self.MIN_H = 120
        self.EDGE_DENSITY_THRESHOLD = 0.003
        self.ENTROPY_THRESHOLD = 3.0
        
        # Permissive fallback parameters (for difficult visuals like Gantt)
        self.MIN_AREA_PERMISSIVE = 8000
        self.EDGE_DENSITY_THRESHOLD_PERMISSIVE = 0.0015
        self.ENTROPY_THRESHOLD_PERMISSIVE = 2.5
        self.MIN_W_PERMISSIVE = 80
        self.MIN_H_PERMISSIVE = 60
        
        print("[OK] ImageDetector initialized")
        print(f"  - DPI: {self.DPI}")
        print(f"  - Min area (strict): {self.MIN_AREA}px")
        print(f"  - Min area (permissive): {self.MIN_AREA_PERMISSIVE}px")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            "model": "opencv-morphological-detection",
            "dpi": self.DPI,
            "min_area_strict": self.MIN_AREA,
            "min_area_permissive": self.MIN_AREA_PERMISSIVE
        }
    
    def detect_images(
        self, 
        page_image: Image.Image,
        page_number: int,
        pdf_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Detect visual elements (charts, graphs, diagrams) on a page
        
        Returns:
            List of detected visuals with bbox and type
        """
        # Convert PIL to OpenCV
        page_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        
        # Get text boxes to mask them out
        text_boxes = []
        if pdf_path:
            text_boxes = self._get_text_boxes(pdf_path, page_number - 1)
        
        # Apply text mask
        masked_gray, mask = self._apply_text_mask(page_bgr, text_boxes)
        
        # Check if page is text-heavy
        if self._page_is_text_heavy(page_bgr, text_boxes, threshold=0.90):
            print(f"    [i] Page {page_number} is >90% text, using permissive mode only")
            boxes = self._detect_visuals_permissive(page_bgr, mask)
        else:
            # Try strict detection first
            boxes = self._detect_visuals_strict(page_bgr, mask)
            
            # Fallback to permissive if nothing found
            if not boxes:
                print(f"    [i] No strict detections, trying permissive mode...")
                boxes = self._detect_visuals_permissive(page_bgr, mask)
        
        if not boxes:
            return []
        
        print(f"    [OK] Detected {len(boxes)} visual(s)")
        
        # Classify each visual
        results = []
        for idx, bbox in enumerate(boxes, start=1):
            x1, y1, x2, y2 = bbox
            crop = page_bgr[y1:y2, x1:x2]
            
            visual_type = self._classify_visual(crop)
            
            results.append({
                "visual_id": f"page_{page_number}_visual_{idx}",
                "bbox": bbox,
                "type": visual_type,
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1)
            })
        
        return results
    
    def _get_text_boxes(self, pdf_path: str, page_index: int) -> List[List[int]]:
        """Get text box coordinates from PDF"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_index)
            raw_blocks = page.get_text("blocks")
            zoom = self.DPI / 72.0
            
            boxes = []
            for b in raw_blocks:
                x1, y1, x2, y2, text = b[0], b[1], b[2], b[3], b[4]
                if text and str(text).strip():
                    boxes.append([
                        int(x1 * zoom), int(y1 * zoom),
                        int(x2 * zoom), int(y2 * zoom)
                    ])
            
            doc.close()
            return boxes
        except Exception as e:
            print(f"    [!] Error getting text boxes: {e}")
            return []
    
    def _apply_text_mask(
        self, 
        page_bgr: np.ndarray, 
        text_boxes: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create mask to exclude text regions"""
        mask = np.ones(page_bgr.shape[:2], dtype=np.uint8) * 255
        
        for (x1, y1, x2, y2) in text_boxes:
            # Add padding around text
            pad_x = int((x2 - x1) * 0.02) + 2
            pad_y = int((y2 - y1) * 0.02) + 2
            
            xa = max(0, x1 - pad_x)
            ya = max(0, y1 - pad_y)
            xb = min(page_bgr.shape[1], x2 + pad_x)
            yb = min(page_bgr.shape[0], y2 + pad_y)
            
            cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)
        
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        return masked, mask
    
    def _page_is_text_heavy(
        self, 
        page_bgr: np.ndarray, 
        text_boxes: List[List[int]], 
        threshold: float = 0.75
    ) -> bool:
        """Check if page is mostly text"""
        h, w = page_bgr.shape[:2]
        page_area = h * w
        text_area = 0
        
        for (x1, y1, x2, y2) in text_boxes:
            text_area += max(0, (x2 - x1) * (y2 - y1))
        
        return (text_area / (page_area + 1e-12)) > threshold
    
    def _detect_visuals_strict(
        self, 
        page_bgr: np.ndarray, 
        mask: np.ndarray
    ) -> List[List[int]]:
        """Strict detection (original behavior)"""
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Adaptive thresholding
        th = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 9
        )
        
        # Morphological operations to merge nearby regions
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        merged = cv2.dilate(th, merge_kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Size filters
            if area < self.MIN_AREA:
                continue
            if w < self.MIN_W or h < self.MIN_H:
                continue
            
            crop = page_bgr[y:y+h, x:x+w]
            
            # Filter out logos and watermarks
            if self._is_logo(crop):
                continue
            if self._is_watermark(crop):
                continue
            
            # Check if it's a real visual
            if self._is_real_visual(crop):
                boxes.append([x, y, x + w, y + h])
        
        # Merge overlapping boxes
        boxes = self._merge_boxes(boxes)
        
        # Sort top to bottom
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        return boxes
    
    def _detect_visuals_permissive(
        self, 
        page_bgr: np.ndarray, 
        mask: np.ndarray
    ) -> List[List[int]]:
        """Permissive detection for difficult visuals (Gantt, faint charts)"""
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        
        # Relax mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        relaxed_mask = cv2.erode(mask, kernel, iterations=1)
        gray = cv2.bitwise_and(gray, gray, mask=relaxed_mask)
        
        # Smaller kernel for preserving smaller shapes
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 41, 7
        )
        
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        merged = cv2.dilate(th, merge_kernel, iterations=1)
        
        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # More lenient size filters
            if area < self.MIN_AREA_PERMISSIVE:
                continue
            if w < self.MIN_W_PERMISSIVE or h < self.MIN_H_PERMISSIVE:
                continue
            
            crop = page_bgr[y:y+h, x:x+w]
            
            # Relaxed logo check
            if self._is_logo(crop) and (w < 220 and h < 220):
                continue
            
            # Relaxed watermark check
            if self._is_watermark(crop) and area > 200000:
                continue
            
            # Permissive visual test
            if self._is_real_visual(
                crop,
                edge_thresh=self.EDGE_DENSITY_THRESHOLD_PERMISSIVE,
                entropy_thresh=self.ENTROPY_THRESHOLD_PERMISSIVE,
                area_thresh=100000
            ):
                boxes.append([x, y, x + w, y + h])
        
        boxes = self._merge_boxes(boxes, iou_thresh=0.1)
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        return boxes
    
    def _is_logo(self, crop: np.ndarray) -> bool:
        """Detect if region is a logo"""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return False
        
        ar = w / h
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        entropy = self._shannon_entropy(gray)
        
        if entropy < 3.0 and (0.5 < ar < 1.8) and (w < 360 and h < 360):
            return True
        
        return False
    
    def _is_watermark(self, crop: np.ndarray) -> bool:
        """Detect if region is a watermark"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        area = crop.shape[0] * crop.shape[1]
        
        if area == 0:
            return False
        
        edge_density = np.sum(edges > 0) / (area + 1e-12)
        
        if edge_density < 0.002 and area > 50000:
            return True
        
        return False
    
    def _is_real_visual(
        self, 
        crop: np.ndarray,
        edge_thresh: float = None,
        entropy_thresh: float = None,
        area_thresh: int = 150000
    ) -> bool:
        """Check if region is a real visual element"""
        if edge_thresh is None:
            edge_thresh = self.EDGE_DENSITY_THRESHOLD
        if entropy_thresh is None:
            entropy_thresh = self.ENTROPY_THRESHOLD
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        area = crop.shape[0] * crop.shape[1]
        
        if area == 0:
            return False
        
        edge_density = np.sum(edges > 0) / (area + 1e-12)
        entropy = self._shannon_entropy(gray)
        
        if (edge_density > edge_thresh) or (entropy > entropy_thresh) or (area > area_thresh):
            return True
        
        return False
    
    def _shannon_entropy(self, gray: np.ndarray) -> float:
        """Calculate Shannon entropy of grayscale image"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        probs = hist / (hist.sum() + 1e-12)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def _classify_visual(self, crop: np.ndarray) -> str:
        """Classify visual type"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100,
            minLineLength=100, maxLineGap=20
        )
        
        if lines is not None:
            horiz = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 5)
            vert = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 5)
            
            if horiz and vert:
                return "chart"
            elif horiz > 5:
                return "gantt"
            return "diagram"
        
        return "image"
    
    def _merge_boxes(
        self, 
        boxes: List[List[int]], 
        iou_thresh: float = 0.15
    ) -> List[List[int]]:
        """Merge overlapping boxes"""
        if not boxes:
            return []
        
        boxes_np = np.array(boxes)
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        idxs = list(range(len(boxes)))
        keep = []
        
        while idxs:
            i = idxs.pop(0)
            bx = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            keep.append(bx)
            
            remove = []
            for j in idxs:
                # Calculate IoU
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                union = areas[i] + areas[j] - inter
                iou = inter / (union + 1e-12)
                
                if iou > iou_thresh:
                    remove.append(j)
                    # Expand bounding box
                    keep[-1][0] = min(keep[-1][0], int(x1[j]))
                    keep[-1][1] = min(keep[-1][1], int(y1[j]))
                    keep[-1][2] = max(keep[-1][2], int(x2[j]))
                    keep[-1][3] = max(keep[-1][3], int(y2[j]))
            
            idxs = [k for k in idxs if k not in remove]
        
        return keep