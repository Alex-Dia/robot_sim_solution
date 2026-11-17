"""Vision helpers for the 3006ICT group project.

The runtime controller uses target-specific ORB + RANSAC recognition.
The mission already names the requested target, so the vision task is binary:
"is this requested target visible at the current station?"

The matcher reads only the requested target reference image and live RGB frames;
it does not use Webots world data or target-to-station metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class MatchResult:
    is_match: bool
    good_matches: int
    inliers: int
    inlier_ratio: float
    score: float
    geometry_valid: bool
    median_distance: float
    homography: Optional[np.ndarray] = None


@dataclass
class ClassificationResult:
    best_label: str
    best_score: float
    second_label: str
    second_score: float
    margin: float
    scores: dict[str, float]
    best_scale: int
    best_flipped: bool
    best_location: tuple[int, int]


class TargetMatcher:
    """ORB + RANSAC match for the requested mission target.

    The robot first moves to a consistent inspection pose, so the poster occupies
    a predictable portion of the 160x120 camera image.  Matching the reference at
    a similar pixel scale gives ORB substantially more stable correspondences.
    """

    def __init__(
        self,
        root: Path,
        target_label: str,
        reference_size: int = 64,
        min_good_matches: int = 18,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.40,
        ratio_test: float = 0.75,
    ) -> None:
        self.root = Path(root)
        self.target_label = target_label
        self.reference_size = int(reference_size)
        self.min_good_matches = int(min_good_matches)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ratio_test = float(ratio_test)

        reference_path = self.root / "textures" / f"target_{target_label}.png"
        if not reference_path.exists():
            raise FileNotFoundError(f"Target reference not found: {reference_path}")

        reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference_bgr is None:
            raise RuntimeError(f"Could not load target reference: {reference_path}")

        self.reference_bgr = cv2.resize(
            reference_bgr,
            (self.reference_size, self.reference_size),
            interpolation=cv2.INTER_AREA,
        )
        self.reference_gray = cv2.cvtColor(self.reference_bgr, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.15,
            nlevels=8,
            edgeThreshold=5,
            patchSize=15,
            fastThreshold=7,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.reference_gray, None)
        if self.ref_descriptors is None or len(self.ref_keypoints) < 10:
            raise RuntimeError(f"Not enough ORB features in reference target: {target_label}")

    def _geometry_is_valid(self, homography: Optional[np.ndarray], frame_shape: tuple[int, int]) -> bool:
        if homography is None or not np.isfinite(homography).all():
            return False
        h, w = frame_shape
        s = float(self.reference_size - 1)
        corners = np.float32([[0, 0], [s, 0], [s, s], [0, s]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.isfinite(projected).all():
            return False
        contour = projected.astype(np.float32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        frame_area = float(max(1, h * w))
        area_ratio = area / frame_area
        return 0.02 <= area_ratio <= 2.0

    def match(self, frame_bgr: np.ndarray) -> MatchResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return MatchResult(False, 0, 0, 0.0, 0.0, False, float("inf"), None)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self.orb.detectAndCompute(gray, None)
        if frame_descriptors is None or len(frame_keypoints) < 4:
            return MatchResult(False, 0, 0, 0.0, 0.0, False, float("inf"), None)

        pairs = self.matcher.knnMatch(self.ref_descriptors, frame_descriptors, k=2)
        good = [m for m, n in pairs if m.distance < self.ratio_test * n.distance]
        if not good:
            return MatchResult(False, 0, 0, 0.0, 0.0, False, float("inf"), None)

        median_distance = float(np.median([m.distance for m in good]))
        if len(good) < 4:
            return MatchResult(False, len(good), 0, 0.0, 0.0, False, median_distance, None)

        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inliers / max(1, len(good))
        geometry_valid = self._geometry_is_valid(homography, gray.shape)
        score = float(inliers * inlier_ratio)

        is_match = (
            len(good) >= self.min_good_matches
            and inliers >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
        )
        return MatchResult(
            is_match=is_match,
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            score=score,
            geometry_valid=geometry_valid,
            median_distance=median_distance,
            homography=homography,
        )

    def draw_debug(self, frame_bgr: np.ndarray, result: MatchResult) -> np.ndarray:
        output = frame_bgr.copy()
        if result.homography is not None and result.geometry_valid:
            s = float(self.reference_size - 1)
            corners = np.float32([[0, 0], [s, 0], [s, s], [0, s]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, result.homography).astype(np.int32)
            cv2.polylines(output, [projected], True, (0, 255, 0), 2, cv2.LINE_AA)
        label = (
            f"ORB match={result.is_match} good={result.good_matches} "
            f"inliers={result.inliers} ratio={result.inlier_ratio:.2f}"
        )
        cv2.putText(output, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        return output


class TemplateClassifier:
    """Classify the station poster using multi-scale colour template matching.

    Why this is included:
    the live e-puck camera is only 160x120.  At that resolution the poster can
    contain too few stable ORB keypoints even when the correct target is plainly
    visible.  Colour template matching preserves the actual appearance of the
    supplied target image and works well on much smaller renderings.

    Both the normal and horizontally flipped reference are tested.  This makes
    the comparison robust to UV orientation of the textured face; it does not
    use simulator ground truth or station-to-target metadata.
    """

    def __init__(
        self,
        root: Path,
        labels: Sequence[str],
        scales: Sequence[int] = (28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96),
    ) -> None:
        self.root = Path(root)
        self.labels = tuple(labels)
        self.scales = tuple(int(s) for s in scales)
        self.templates: dict[str, list[tuple[int, bool, np.ndarray]]] = {}

        for label in self.labels:
            path = self.root / "textures" / f"target_{label}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not load target reference: {path}")

            variants: list[tuple[int, bool, np.ndarray]] = []
            for flipped, base in ((False, image), (True, cv2.flip(image, 1))):
                for size in self.scales:
                    template = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)
                    variants.append((size, flipped, template))
            self.templates[label] = variants

    def classify(self, frame_bgr: np.ndarray) -> ClassificationResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("camera frame is empty")

        frame_h, frame_w = frame_bgr.shape[:2]
        scores: dict[str, float] = {}
        details: dict[str, tuple[int, bool, tuple[int, int]]] = {}

        for label in self.labels:
            best_score = -1.0
            best_scale = 0
            best_flipped = False
            best_location = (0, 0)

            for size, flipped, template in self.templates[label]:
                if size >= frame_h or size >= frame_w:
                    continue
                response = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
                _, max_value, _, max_location = cv2.minMaxLoc(response)
                if max_value > best_score:
                    best_score = float(max_value)
                    best_scale = size
                    best_flipped = flipped
                    best_location = max_location

            scores[label] = best_score
            details[label] = (best_scale, best_flipped, best_location)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_label, second_score = ranked[1] if len(ranked) > 1 else ("", -1.0)
        best_scale, best_flipped, best_location = details[best_label]
        return ClassificationResult(
            best_label=best_label,
            best_score=float(best_score),
            second_label=second_label,
            second_score=float(second_score),
            margin=float(best_score - second_score),
            scores=scores,
            best_scale=best_scale,
            best_flipped=best_flipped,
            best_location=best_location,
        )

    def draw_debug(self, frame_bgr: np.ndarray, result: ClassificationResult) -> np.ndarray:
        output = frame_bgr.copy()
        x, y = result.best_location
        s = result.best_scale
        if s > 0:
            cv2.rectangle(output, (x, y), (x + s, y + s), (0, 255, 0), 1)
        text1 = f"Template: {result.best_label} score={result.best_score:.3f} margin={result.margin:.3f}"
        text2 = f"2nd: {result.second_label} {result.second_score:.3f} scale={s} flip={result.best_flipped}"
        cv2.putText(output, text1, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(output, text2, (3, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 255), 1, cv2.LINE_AA)
        return output
