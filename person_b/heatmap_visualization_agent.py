import numpy as np
import matplotlib.pyplot as plt
import os

class HeatmapVisualizationAgent:
    """
    Person B: Heatmap Visualization Agent
    Purpose: Render fault heatmaps on a vehicle diagram.
    """

    def __init__(self):
        self.vehicle_diagram = None

    def load_vehicle_diagram(self):
        """Load base image or generate placeholder."""
        # Generate a simple placeholder diagram (white rectangle with wheels)
        h, w = 400, 400
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw "Vehicle" box
        img[100:300, 150:250] = [200, 200, 200]
        # Wheels
        img[120:160, 130:150] = [50, 50, 50]
        img[120:160, 250:270] = [50, 50, 50]
        img[240:280, 130:150] = [50, 50, 50]
        img[240:280, 250:270] = [50, 50, 50]
        
        self.vehicle_diagram = img
        return img

    def color_map_normalize(self, heatmap):
        """Normalize heatmap to [0, 1]."""
        mn = np.min(heatmap)
        mx = np.max(heatmap)
        if mx - mn < 1e-9:
            return np.zeros_like(heatmap)
        return (heatmap - mn) / (mx - mn)

    def resize_heatmap(self, heatmap, target_shape):
        """Resize heatmap to match diagram."""
        # Simple nearest neighbor or block expansion
        h, w = target_shape[:2]
        gh, gw = heatmap.shape
        
        # Kronecker product for scaling up (if integer scale)
        # Or scipy.ndimage.zoom
        scale_h = h // gh
        scale_w = w // gw
        
        resized = np.kron(heatmap, np.ones((scale_h, scale_w)))
        
        # Pad if needed
        rh, rw = resized.shape
        pad_h = h - rh
        pad_w = w - rw
        if pad_h > 0 or pad_w > 0:
            resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode='edge')
            
        return resized

    def overlay_heatmap_on_diagram(self, heatmap):
        """Overlay heatmap onto diagram."""
        if self.vehicle_diagram is None:
            self.load_vehicle_diagram()
            
        base = self.vehicle_diagram.copy()
        h, w, _ = base.shape
        
        # Normalize and resize
        norm_map = self.color_map_normalize(heatmap)
        resized_map = self.resize_heatmap(norm_map, (h, w))
        
        # Create RGB heatmap (Red channel intensity)
        overlay = np.zeros_like(base)
        overlay[:, :, 0] = (resized_map * 255).astype(np.uint8) # Red
        
        # Alpha blend
        alpha = 0.5
        mask = resized_map > 0.1 # Only overlay where there is heat
        
        blended = base.copy()
        blended[mask] = (1 - alpha) * base[mask] + alpha * overlay[mask]
        
        return blended.astype(np.uint8)

    def export_heatmap_png(self, heatmap, output_path):
        """Save heatmap.png."""
        final_img = self.overlay_heatmap_on_diagram(heatmap)
        plt.imsave(output_path, final_img)

    def generate_visualization(self, heatmap_array, output_path):
        """Main entry point."""
        self.export_heatmap_png(heatmap_array, output_path)
