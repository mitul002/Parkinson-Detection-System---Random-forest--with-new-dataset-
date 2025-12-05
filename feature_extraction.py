# -*- coding: utf-8 -*-
"""
Advanced Feature Extraction for Parkinson's Disease Detection from Spiral Drawings
Comprehensive feature extraction combining handcrafted and CNN features
"""

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops, shannon_entropy
from skimage.util import img_as_ubyte, img_as_bool
from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing
from scipy.signal import savgol_filter
import warnings
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models

warnings.filterwarnings("ignore", category=RuntimeWarning)

class SpiralFeatureExtractor:
    """
    Advanced feature extractor for spiral drawing analysis
    Combines handcrafted geometric features with CNN features
    """
    
    def __init__(self, use_cnn=True, device='cpu'):
        self.use_cnn = use_cnn
        self.device = device
        
        # Initialize CNN model if requested
        if self.use_cnn:
            self.cnn_model = self._setup_cnn_model()
            self.cnn_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _setup_cnn_model(self):
        """Initialize and configure the CNN model for feature extraction"""
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()  # Remove final classification layer
        model = model.to(self.device)
        model.eval()
        return model
    
    @staticmethod
    def safe_div(a, b, default=np.nan):
        """Safe division with default value for zero division"""
        return a / b if (b is not None and b != 0) else default
    
    @staticmethod
    def iqr_trim(values, k=1.5):
        """Remove outliers using IQR method"""
        if values.size == 0:
            return values
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low, high = q1 - k*iqr, q3 + k*iqr
        return values[(values >= low) & (values <= high)]
    
    def keep_largest_component(self, binary):
        """Keep only the largest connected component"""
        lbl = label(binary, connectivity=2)
        if lbl.max() == 0:
            return binary
        regions = regionprops(lbl)
        largest = max(regions, key=lambda r: r.area).label
        return lbl == largest
    
    def auto_binarize(self, gray):
        """Automatic binarization with polarity detection"""
        # CLAHE preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        
        # Try both polarities
        _, th_dark = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, th_light = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Score each polarity
        def score(b):
            p = (b == 255).mean()
            return -abs(p - 0.18)  # Prefer ~18% foreground coverage
        
        cand = th_dark if score(th_dark) >= score(th_light) else th_light
        
        # Fallback to adaptive if coverage is poor
        fg_ratio = (cand == 255).mean()
        if fg_ratio < 0.01 or fg_ratio > 0.6:
            cand = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 35, 5)
        
        # Ensure foreground = True
        if cand.mean() > 127:
            cand = 255 - cand
        
        binary = (cand // 255).astype(bool)
        return binary
    
    def morph_clean(self, bw, open_iters=1, close_iters=1):
        """Morphological cleaning of binary image"""
        if open_iters > 0:
            bw = binary_opening(bw, iterations=open_iters)
        if close_iters > 0:
            bw = binary_closing(bw, iterations=close_iters)
        bw = remove_small_objects(bw, min_size=64)
        return bw
    
    def get_main_contour(self, bw_uint8):
        """Extract the largest contour from binary image"""
        contours, _ = cv2.findContours(bw_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    
    def smooth_contour(self, cnt, window=21, poly=3):
        """Smooth contour using Savitzky-Golay filter"""
        arr = cnt[:, 0, :].astype(np.float64)
        n = len(arr)
        if n < window:
            window = max(5, (n // 2) * 2 + 1)
        if window < 5:
            return arr
        
        x, y = arr[:, 0], arr[:, 1]
        try:
            xs = savgol_filter(x, window_length=window, polyorder=min(poly, window-2), mode='interp')
            ys = savgol_filter(y, window_length=window, polyorder=min(poly, window-2), mode='interp')
            return np.stack([xs, ys], axis=1)
        except:
            return arr
    
    def curvature_stats(self, path):
        """Calculate curvature statistics"""
        if path.shape[0] < 5:
            return dict(k_mean=np.nan, k_std=np.nan, k_p95=np.nan)
        
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx*dx + dy*dy)**1.5
        
        kappa = np.zeros_like(dx)
        mask = denom > 1e-9
        kappa[mask] = np.abs(dx[mask]*ddy[mask] - dy[mask]*ddx[mask]) / denom[mask]
        
        kappa_f = self.iqr_trim(kappa[~np.isnan(kappa)])
        if kappa_f.size == 0:
            return dict(k_mean=np.nan, k_std=np.nan, k_p95=np.nan)
        
        return dict(
            k_mean=float(np.mean(kappa_f)),
            k_std=float(np.std(kappa_f)),
            k_p95=float(np.percentile(kappa_f, 95))
        )
    
    def stroke_width_stats(self, bin_uint8):
        """Calculate stroke width statistics using distance transform"""
        dist = distance_transform_edt(bin_uint8 > 0)
        skel = skeletonize(bin_uint8 > 0).astype(np.uint8)
        thickness = dist * skel * 2.0
        vals = thickness[thickness > 0].astype(np.float64)
        vals = self.iqr_trim(vals)
        
        if vals.size == 0:
            return dict(sw_mean=np.nan, sw_std=np.nan, sw_p95=np.nan)
        
        return dict(
            sw_mean=float(np.mean(vals)),
            sw_std=float(np.std(vals)),
            sw_p95=float(np.percentile(vals, 95))
        )
    
    def radial_features(self, path):
        """Extract radial features from path"""
        cx, cy = np.mean(path[:, 0]), np.mean(path[:, 1])
        dx = path[:, 0] - cx
        dy = path[:, 1] - cy
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        theta_unw = np.unwrap(theta)
        
        # Calculate wobble and turns
        dtheta = np.diff(theta_unw)
        wobble = float(np.std(dtheta)) if dtheta.size else np.nan
        turns = float(np.abs(theta_unw[-1] - theta_unw[0]) / (2*np.pi)) if theta_unw.size > 1 else np.nan
        
        r_f = self.iqr_trim(r)
        return r, theta_unw, dict(
            radial_mean=float(np.mean(r_f)) if r_f.size else np.nan,
            radial_std=float(np.std(r_f)) if r_f.size else np.nan,
            wobble=wobble,
            turns=turns
        )
    
    def fit_archimedean_spiral(self, r, theta_unw):
        """Fit Archimedean spiral model r = a + b*theta"""
        if len(r) < 10:
            return dict(a=np.nan, b=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan)
        
        X = np.vstack([np.ones_like(theta_unw), theta_unw]).T
        try:
            beta, *_ = np.linalg.lstsq(X, r, rcond=None)
            a, b = beta
            r_pred = a + b*theta_unw
            resid = r - r_pred
            rmse = float(np.sqrt(np.mean(resid**2)))
            mae = float(np.mean(np.abs(resid)))
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((r - np.mean(r))**2) + 1e-12
            r2 = float(1 - ss_res/ss_tot)
            return dict(a=float(a), b=float(b), rmse=rmse, mae=mae, r2=r2)
        except:
            return dict(a=np.nan, b=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan)
    
    def radial_error_to_ideal(self, path, a=1.0, b=4.0):
        """Calculate error compared to ideal spiral"""
        cx, cy = np.mean(path[:, 0]), np.mean(path[:, 1])
        dx = path[:, 0] - cx
        dy = path[:, 1] - cy
        r = np.hypot(dx, dy)
        theta = np.unwrap(np.arctan2(dy, dx))
        r_ideal = a + b*theta
        
        # Normalize scales
        med_r = np.median(r) if r.size else 1.0
        if med_r == 0:
            med_r = 1.0
        r_n = r / med_r
        r_i = r_ideal / (np.median(r_ideal) if np.median(r_ideal) != 0 else 1.0)
        diff = r_n - r_i
        
        if diff.size == 0:
            return dict(rad_err_mean=np.nan, rad_err_med=np.nan, rad_err_max=np.nan)
        
        return dict(
            rad_err_mean=float(np.mean(np.abs(diff))),
            rad_err_med=float(np.median(np.abs(diff))),
            rad_err_max=float(np.max(np.abs(diff)))
        )
    
    def contour_metrics(self, cnt):
        """Calculate geometric metrics from contour"""
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = self.safe_div(w, h)
        compact = self.safe_div(peri**2, (4*np.pi*area)) if area > 0 else np.nan
        circularity = self.safe_div(4*np.pi*area, peri**2) if peri > 0 else np.nan
        
        # Symmetry score
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
            radii = np.hypot(cnt[:, 0, 0]-cx, cnt[:, 0, 1]-cy)
            sym = self.safe_div(np.mean(radii), np.std(radii)) if np.std(radii) > 0 else np.nan
        else:
            sym = np.nan
        
        # Stroke length
        pts = cnt[:, 0, :]
        diffs = np.diff(pts, axis=0)
        stroke_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        
        return dict(
            area=float(area),
            perimeter=float(peri),
            aspect_ratio=float(aspect) if aspect == aspect else np.nan,
            compactness=float(compact) if compact == compact else np.nan,
            circularity=float(circularity) if circularity == circularity else np.nan,
            symmetry_score=float(sym) if sym == sym else np.nan,
            stroke_length=round(stroke_len, 2)
        )
    
    def skeleton_graph_features(self, bw_bool):
        """Extract skeleton-based graph features"""
        skel = skeletonize(bw_bool).astype(np.uint8)
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        neigh = cv2.filter2D(skel, -1, kernel)
        
        ep = np.logical_and(skel == 1, neigh == 11)     # endpoints
        bp = np.logical_and(skel == 1, neigh >= 13)     # branch points
        
        return skel, dict(
            skel_endpoints=int(np.count_nonzero(ep)),
            skel_branchpoints=int(np.count_nonzero(bp)),
            skel_length=int(np.count_nonzero(skel))
        )
    
    def fractal_dimension_boxcount(self, bw_bool):
        """Calculate fractal dimension using box counting"""
        skel = skeletonize(bw_bool)
        Z = skel.astype(np.uint8)
        if Z.sum() == 0:
            return np.nan
        
        sizes, counts = [], []
        p = int(np.floor(np.log2(min(Z.shape))))
        
        for k in range(p, 1, -1):
            s = 2**k
            sizes.append(s)
            new_shape = (int(np.ceil(Z.shape[0]/s))*s, int(np.ceil(Z.shape[1]/s))*s)
            pad = np.zeros(new_shape, dtype=np.uint8)
            pad[:Z.shape[0], :Z.shape[1]] = Z
            S = pad.reshape(new_shape[0]//s, s, new_shape[1]//s, s).any(axis=(1,3))
            counts.append(np.count_nonzero(S))
        
        if len(counts) < 2:
            return np.nan
        
        sizes = np.array(sizes, dtype=np.float64)
        counts = np.array(counts, dtype=np.float64) + 1e-9
        x = np.log(1.0/sizes)
        y = np.log(counts)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    
    def extract_cnn_features(self, image_path):
        """Extract CNN features from image"""
        if not self.use_cnn:
            return np.array([])
        
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.cnn_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.cnn_model(img_tensor).cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Warning: CNN feature extraction failed: {e}")
            return np.zeros(512)  # ResNet18 feature size
    
    def extract_handcrafted_features(self, image_path):
        """Extract comprehensive handcrafted features from spiral image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 1) Preprocess → binary (stroke=True)
        bw = self.auto_binarize(img)
        bw = self.morph_clean(bw, open_iters=1, close_iters=1)
        bw = self.keep_largest_component(bw)
        
        bw_bool = bw.astype(bool)
        bw_u8 = (bw_bool.astype(np.uint8)) * 255
        
        # Global entropy
        entropy = float(shannon_entropy(img_as_ubyte(img)))
        
        # 2) Stroke width analysis
        sw = self.stroke_width_stats(bw_u8)
        
        # 3) Contour analysis
        cnt = self.get_main_contour(bw_u8)
        if cnt is None or len(cnt) < 5:
            # Return minimal info
            skel, skf = self.skeleton_graph_features(bw_bool)
            return dict(
                Mean_Thickness=sw['sw_mean'], Std_Thickness=sw['sw_std'], 
                Thickness_P95=sw['sw_p95'], Entropy=round(entropy, 2),
                Skel_Endpoints=skf['skel_endpoints'],
                Skel_Branchpoints=skf['skel_branchpoints'],
                Skel_Length=skf['skel_length'],
                # Fill rest with NaN
                **{k: np.nan for k in ['Loop_Variation', 'Jerkiness', 'Curv_Mean', 'Curv_Std', 
                                      'Curv_P95', 'Path_RMSE', 'Path_MAE', 'Spiral_R2', 
                                      'Spiral_a', 'Spiral_b', 'Rad_Err_Mean', 'Rad_Err_Med', 
                                      'Rad_Err_Max', 'Stroke_Length', 'Aspect_Ratio', 
                                      'Compactness', 'Circularity', 'Symmetry_Score', 
                                      'Radial_Mean', 'Radial_Std', 'Wobble', 'Turns', 
                                      'Fractal_Dim']}
            )
        
        # 4) Smooth path & curvature analysis
        path = self.smooth_contour(cnt, window=21, poly=3)
        curv = self.curvature_stats(path)
        
        # Jerkiness calculation
        v = np.diff(path, axis=0)
        angles = np.arctan2(v[:, 1], v[:, 0])
        dtheta = np.diff(np.unwrap(angles))
        jerkiness = float(np.std(dtheta)) if dtheta.size else np.nan
        
        # Loop variation
        cx, cy = np.mean(path[:, 0]), np.mean(path[:, 1])
        radii = np.hypot(path[:, 0]-cx, path[:, 1]-cy)
        loop_var = float(np.std(radii)) if radii.size else np.nan
        
        # 5) Polar & spiral analysis
        r, theta_unw, radial = self.radial_features(path)
        fit = self.fit_archimedean_spiral(r, theta_unw)
        
        # 6) Error analysis
        rad_err = self.radial_error_to_ideal(path, a=1.0, b=4.0)
        
        # 7) Contour metrics
        cm = self.contour_metrics(cnt)
        
        # 8) Skeleton & fractal analysis
        skel, skf = self.skeleton_graph_features(bw_bool)
        fracdim = self.fractal_dimension_boxcount(bw_bool)
        
        # Assemble features (using consistent naming convention)
        features = dict(
            Mean_Thickness=round(sw['sw_mean'], 3) if sw['sw_mean'] == sw['sw_mean'] else np.nan,
            Std_Thickness=round(sw['sw_std'], 3) if sw['sw_std'] == sw['sw_std'] else np.nan,
            Thickness_P95=round(sw['sw_p95'], 3) if sw['sw_p95'] == sw['sw_p95'] else np.nan,
            Entropy=round(entropy, 3),
            
            Loop_Variation=round(loop_var, 3) if loop_var == loop_var else np.nan,
            Jerkiness=round(jerkiness, 5) if jerkiness == jerkiness else np.nan,
            Curv_Mean=round(curv['k_mean'], 5) if curv['k_mean'] == curv['k_mean'] else np.nan,
            Curv_Std=round(curv['k_std'], 5) if curv['k_std'] == curv['k_std'] else np.nan,
            Curv_P95=round(curv['k_p95'], 5) if curv['k_p95'] == curv['k_p95'] else np.nan,
            
            Path_RMSE=round(fit['rmse'], 4) if fit['rmse'] == fit['rmse'] else np.nan,
            Path_MAE=round(fit['mae'], 4) if fit['mae'] == fit['mae'] else np.nan,
            Spiral_R2=round(fit['r2'], 4) if fit['r2'] == fit['r2'] else np.nan,
            Spiral_a=round(fit['a'], 5) if fit['a'] == fit['a'] else np.nan,
            Spiral_b=round(fit['b'], 5) if fit['b'] == fit['b'] else np.nan,
            
            Rad_Err_Mean=round(rad_err['rad_err_mean'], 5) if rad_err['rad_err_mean'] == rad_err['rad_err_mean'] else np.nan,
            Rad_Err_Med=round(rad_err['rad_err_med'], 5) if rad_err['rad_err_med'] == rad_err['rad_err_med'] else np.nan,
            Rad_Err_Max=round(rad_err['rad_err_max'], 5) if rad_err['rad_err_max'] == rad_err['rad_err_max'] else np.nan,
            
            Stroke_Length=cm['stroke_length'],
            Aspect_Ratio=round(cm['aspect_ratio'], 4) if cm['aspect_ratio'] == cm['aspect_ratio'] else np.nan,
            Compactness=round(cm['compactness'], 4) if cm['compactness'] == cm['compactness'] else np.nan,
            Circularity=round(cm['circularity'], 4) if cm['circularity'] == cm['circularity'] else np.nan,
            Symmetry_Score=round(cm['symmetry_score'], 4) if cm['symmetry_score'] == cm['symmetry_score'] else np.nan,
            
            Radial_Mean=round(radial['radial_mean'], 5) if radial['radial_mean'] == radial['radial_mean'] else np.nan,
            Radial_Std=round(radial['radial_std'], 5) if radial['radial_std'] == radial['radial_std'] else np.nan,
            Wobble=round(radial['wobble'], 5) if radial['wobble'] == radial['wobble'] else np.nan,
            Turns=round(radial['turns'], 3) if radial['turns'] == radial['turns'] else np.nan,
            
            # Use capitalized naming for consistency with trained model
            Skel_Endpoints=skf['skel_endpoints'],
            Skel_Branchpoints=skf['skel_branchpoints'],
            Skel_Length=skf['skel_length'],
            Fractal_Dim=round(fracdim, 5) if fracdim == fracdim else np.nan,
        )
        
        return features
    
    def extract_features(self, image_path):
        """Extract both CNN and handcrafted features"""
        cnn_features = self.extract_cnn_features(image_path) if self.use_cnn else np.array([])
        handcrafted_features = self.extract_handcrafted_features(image_path)
        
        if handcrafted_features is None:
            return None
        
        return {
            'cnn_features': cnn_features,
            'handcrafted_features': handcrafted_features
        }

# Usage example:
if __name__ == "__main__":
    extractor = SpiralFeatureExtractor(use_cnn=True, device='cpu')
    features = extractor.extract_features("path_to_spiral_image.jpg")
    if features:
        print("CNN features shape:", features['cnn_features'].shape)
        print("Handcrafted features:", len(features['handcrafted_features']))