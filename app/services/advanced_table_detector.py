"""
Advanced Table Detection with Multi-Scale Strategy
Based on microsoft/table-transformer-detection
"""

import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from typing import List, Dict, Any, Tuple
import math

class AdvancedTableDetector:
    """
    Multi-scale table detection with aggressive fallback strategies
    """
    
    def __init__(self):
        self.MODEL_NAME = "microsoft/table-transformer-detection"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = None
        self.model = None
        self.loaded = False
        
        # Detection parameters
        self.base_confidence = 0.25  # Ultra-low base threshold
        self.nms_threshold = 0.5     # Higher to preserve separate tables
        
        self._load_model()
    
    def _load_model(self):
        """Load table detection model"""
        try:
            print(f"[i] Loading table detection model: {self.MODEL_NAME}")
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
            self.model = TableTransformerForObjectDetection.from_pretrained(self.MODEL_NAME)
            self.model.to(self.DEVICE)
            self.model.eval()
            self.loaded = True
            print(f"[OK] Table detection model loaded on {self.DEVICE}")
        except Exception as e:
            print(f"[ERROR] Failed to load table detection model: {e}")
            self.loaded = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            "model": self.MODEL_NAME,
            "loaded": self.loaded,
            "device": self.DEVICE,
            "confidence_threshold": self.base_confidence,
            "nms_threshold": self.nms_threshold
        }
    
    def detect_tables(
        self, 
        page_image: Image.Image, 
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Detect tables using multi-scale aggressive strategy
        
        Returns:
            List of detected tables with bbox and confidence
        """
        if not self.loaded:
            print(f"[!] Table detection model not loaded")
            return []
        
        print(f"  Starting multi-scale detection...")
        all_detections = []
        
        # Pass 1: Standard detection
        processed = self._preprocess_standard(page_image)
        tables_pass1 = self._run_detection(processed, self.base_confidence)
        all_detections.extend(tables_pass1)
        print(f"    Pass 1 (standard): {len(tables_pass1)} tables")
        
        # Pass 2: High contrast for faint tables
        high_contrast = self._preprocess_high_contrast(page_image)
        tables_pass2 = self._run_detection(high_contrast, self.base_confidence * 0.8)
        all_detections.extend(tables_pass2)
        print(f"    Pass 2 (high contrast): {len(tables_pass2)} tables")
        
        # Pass 3: Grayscale for structure
        grayscale = self._preprocess_grayscale(page_image)
        tables_pass3 = self._run_detection(grayscale, self.base_confidence * 0.9)
        all_detections.extend(tables_pass3)
        print(f"    Pass 3 (grayscale): {len(tables_pass3)} tables")
        
        print(f"    Total raw detections: {len(all_detections)}")
        
        if not all_detections:
            # Fallback: Ultra-aggressive mode
            print(f"    [!] No tables found, trying ultra-aggressive mode...")
            ultra_aggressive = self._preprocess_edges(page_image)
            tables_fallback = self._run_detection(ultra_aggressive, 0.15)
            all_detections.extend(tables_fallback)
            print(f"    Fallback: {len(tables_fallback)} tables")
        
        if not all_detections:
            return []
        
        # Apply NMS
        tables = self._non_max_suppression(all_detections)
        print(f"    After NMS: {len(tables)} tables")
        
        # Filter tiny tables
        filtered = self._filter_small_tables(tables, page_image)
        print(f"    After filtering: {len(filtered)} tables")
        
        # Sort top to bottom
        filtered.sort(key=lambda t: t['bbox'][1])
        
        return filtered
    
    def _preprocess_standard(self, image: Image.Image) -> Image.Image:
        """Standard preprocessing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.10)
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.05)
        
        return enhanced
    
    def _preprocess_high_contrast(self, image: Image.Image) -> Image.Image:
        """High contrast for faint tables"""
        rgb = image.convert('RGB')
        enhancer = ImageEnhance.Contrast(rgb)
        return enhancer.enhance(1.4)
    
    def _preprocess_grayscale(self, image: Image.Image) -> Image.Image:
        """Grayscale for structure"""
        gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        gray_enhanced = enhancer.enhance(1.3)
        return gray_enhanced.convert('RGB')
    
    def _preprocess_edges(self, image: Image.Image) -> Image.Image:
        """Edge detection for ultra-difficult cases"""
        edges = image.filter(ImageFilter.FIND_EDGES)
        return edges.convert('RGB')
    
    def _run_detection(
        self, 
        image: Image.Image, 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Run single detection pass"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]]).to(self.DEVICE)
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=threshold, 
                target_sizes=target_sizes
            )[0]
            
            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if int(label.item()) == 0:  # table class
                    tables.append({
                        "bbox": [float(x) for x in box.tolist()],
                        "confidence": float(score.item())
                    })
            
            return tables
            
        except Exception as e:
            print(f"[!] Detection error: {e}")
            return []
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _non_max_suppression(
        self, 
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping detections
        Higher threshold (0.5) to preserve separate tables
        """
        if not detections:
            return []
        
        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        kept = []
        
        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            
            # Remove only heavily overlapping boxes
            sorted_dets = [
                det for det in sorted_dets 
                if self._compute_iou(best["bbox"], det["bbox"]) < self.nms_threshold
            ]
        
        return kept
    
    def _filter_small_tables(
        self, 
        tables: List[Dict[str, Any]], 
        page_image: Image.Image
    ) -> List[Dict[str, Any]]:
        """Filter out very small boxes"""
        img_area = page_image.width * page_image.height
        min_area = img_area * 0.0002  # 0.02% of page
        
        filtered = []
        for t in tables:
            x1, y1, x2, y2 = t["bbox"]
            area = (x2 - x1) * (y2 - y1)
            
            if area >= min_area:
                filtered.append(t)
            else:
                print(f"      [i] Filtered tiny box: area={area:.0f}px")
        
        return filtered