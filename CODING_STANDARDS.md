# Engineering Standards and Architecture: RGB-to-Skeleton MediaPipe Pipeline

**Version**: 2.0.0  
**Last Updated**: May 14, 2026  
**Status**: Q1 Research Paper Standard  
**Audience**: Research Engineers, Data Scientists, ML Engineers

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architectural Philosophy](#2-architectural-philosophy)
3. [Strict Single Responsibility Principle (SRP)](#3-strict-single-responsibility-principle-srp)
4. [Directory Structure and Module Organization](#4-directory-structure-and-module-organization)
5. [Naming Conventions](#5-naming-conventions)
6. [Core Architecture Patterns](#6-core-architecture-patterns)
7. [Python Code Standards](#7-python-code-standards)
8. [Data Structure Specifications](#8-data-structure-specifications)
9. [Configuration Management Framework](#9-configuration-management-framework)
10. [Error Handling and Logging](#10-error-handling-and-logging)
11. [Testing and Validation Strategy](#11-testing-and-validation-strategy)
12. [SOLID Design Principles](#12-solid-design-principles)
13. [Scalability and Maintainability](#13-scalability-and-maintainability)
14. [Anti-Patterns and What to Avoid](#14-anti-patterns-and-what-to-avoid)
15. [Research Paper Reproducibility](#15-research-paper-reproducibility)
16. [Contributor Guidelines](#16-contributor-guidelines)
17. [Quality Assurance Checklist](#17-quality-assurance-checklist)

---

## 1. Introduction

The RGB-to-Skeleton MediaPipe Pipeline (`rgb-to-skeleton-mediapipe`) is a high-performance data processing system designed to extract standardized skeleton keypoints from sign language video data (BISINDO - Indonesian Sign Language). The project integrates multiple state-of-the-art computer vision libraries (MediaPipe Holistic, OpenCV) and produces deterministic, reproducible outputs across multiple serialization formats (Pickle, Excel).

### Project Scope

**In Scope**: Backend Python pipeline, configuration management, data extraction, validation, and serialization.

**Out of Scope**: Research notebooks (`notebooks/`), exploratory analysis scripts, and visualization utilities.

### Design Objectives

- **Reproducibility**: Every extraction must be deterministic and independently verifiable
- **Maintainability**: Clean separation of concerns enabling future modifications without affecting core logic
- **Scalability**: Support for multiple extractor types, keypoint configurations, and output formats
- **Research Integrity**: Configuration-driven approach suitable for academic publication and peer review
- **Type Safety**: Compile-time verification of data contracts across all modules

---

## 2. Architectural Philosophy

This project implements a **Layered Architecture Pattern** with strict **Separation of Concerns (SoC)** across five distinct layers:

### 2.1 Five-Layer Architecture Model

```
┌────────────────────────────────────────────────────────┐
│ Layer 1: User Interface (CLI)                           │
│ Responsibility: Parse arguments, route execution        │
├────────────────────────────────────────────────────────┤
│ Layer 2: Orchestration & Workflow                       │
│ Responsibility: Coordinate subsystems, manage routing   │
├────────────────────────────────────────────────────────┤
│ Layer 3: Feature Processing & Data Transformation      │
│ ├─ Extractor: MediaPipe integration                    │
│ ├─ Processor: Keypoint selection & validation          │
│ └─ Converter: Multi-format serialization               │
├────────────────────────────────────────────────────────┤
│ Layer 4: Configuration & Metadata                       │
│ Responsibility: Define WHAT to process, not HOW        │
├────────────────────────────────────────────────────────┤
│ Layer 5: Infrastructure & Utilities                     │
│ Responsibility: Shared helpers, logging, error mgmt    │
└────────────────────────────────────────────────────────┘
```

### 2.2 Core Design Principles

1. **Single Responsibility Principle (SRP)**
   - Each module has exactly one reason to change
   - `Holistic86Extractor`: Extraction only, not serialization
   - `KeypointSelector`: Selection and formatting, not validation
   - `PickleConverter`: Pickle format handling, not extraction

2. **Inversion of Control (IoC)**
   - Configuration defines what to extract (not hardcoded)
   - Selection indices centralized in `config/keypoint_layout.py`
   - Processors use configuration, not vice versa

3. **Separation of Configuration from Logic**
   - **WHAT**: Defined in `config/` (keypoint indices, paths, settings)
   - **HOW**: Implemented in `processor/`, `extractor/`, `converter/`
   - **WHERE**: Orchestrated in `core/pipeline.py`

4. **Composition Over Inheritance**
   - `SkeletonPipeline` composes extractors, processors, and converters
   - No deep inheritance hierarchies
   - Flexible strategy pattern for converters

### 2.3 Key Architectural Decisions

| Decision | Rationale | Implication |
|----------|-----------|------------|
| Centralized keypoint config | Reproducibility, single modification point | Researchers can change landmarks without touching extraction code |
| Separate KeypointSelector | Isolates selection logic from model | Framework-agnostic selection (can swap MediaPipe for other models) |
| Strategy pattern for converters | Easy to add new formats | No pipeline modification needed for new output types |
| Explicit configuration files | Research integrity | Full audit trail of what configuration was used |

---

## 3. Strict Single Responsibility Principle (SRP)

This project enforces **strict SRP** at three levels: package, class, and function.

### 3.1 Package-Level Responsibilities

Each top-level package has exactly one reason to change:

#### `src/config/` — Configuration Definitions
```
Responsibility: Define system parameters and constants
Why separate: 
  ✓ Researchers modify configuration without touching code logic
  ✓ Version control tracks configuration changes explicitly
  ✓ Single source of truth for all settings

What belongs here:
  ✓ MEDIAPIPE_CONFIG dictionary
  ✓ KEYPOINT_LAYOUT and selection indices
  ✓ Directory paths and environment settings
  ✓ ID mapping tables (Person, Sentence, Repetition)
  
What does NOT belong here:
  ✗ Class instantiation
  ✗ Business logic or algorithms
  ✗ Experimental features or feature flags
```

#### `src/core/` — Pipeline Orchestration
```
Responsibility: Coordinate extraction, processing, and serialization workflow
Why separate:
  ✓ Single entry point for batch processing
  ✓ Manages error recovery and progress reporting
  ✓ Decouples CLI from internal pipeline logic

What belongs here:
  ✓ SkeletonPipeline orchestration class
  ✓ Video file discovery and routing
  ✓ Split mapping management (train/test assignment)
  ✓ Video ID parsing and metadata extraction
  ✓ Top-level error handling and logging

What does NOT belong here:
  ✗ MediaPipe model instantiation details
  ✗ Format-specific serialization logic
  ✗ Keypoint validation rules
```

#### `src/extractor/` — Feature Extraction
```
Responsibility: Integrate computer vision model for landmark detection
Why separate:
  ✓ Model initialization and inference isolated from rest of pipeline
  ✓ Easy to benchmark or swap with other pose estimation models
  ✓ Clean interface: video_path → numpy array

What belongs here:
  ✓ Holistic86Extractor class
  ✓ Frame-level processing
  ✓ BGR-to-RGB color space conversion
  ✓ Video metadata handling (rotation, codec)
  
What does NOT belong here:
  ✗ Keypoint selection indices (that's config/)
  ✗ Landmark formatting (that's processor/)
  ✗ File serialization (that's converter/)
```

#### `src/processor/` — Data Processing & Validation
```
Responsibility: Transform raw landmarks into structured keypoints
Why separate:
  ✓ Selection algorithm isolated for testing and modification
  ✓ Validation logic reusable across different extractors
  ✓ Clear data contract: landmarks → validated keypoints

What belongs here:
  ✓ KeypointSelector: landmark selection and formatting
  ✓ Validators: shape and value validation
  ✓ Data transformation utilities (normalization, filtering)
  
What does NOT belong here:
  ✗ Model inference (that's extractor/)
  ✗ Output format selection (that's converter/)
  ✗ Configuration definitions (that's config/)
```

#### `src/converter/` — Output Serialization
```
Responsibility: Convert numpy arrays to target storage format
Why separate:
  ✓ Adding new format requires only new converter subclass
  ✓ Each format's I/O logic encapsulated
  ✓ Framework-agnostic: could export to HDF5, NetCDF, Parquet, etc.

What belongs here:
  ✓ PickleConverter: Pickle-specific serialization
  ✓ ExcelConverter: Excel-specific tabularization
  ✓ Format-specific validation and metadata

What does NOT belong here:
  ✗ Keypoint extraction (that's extractor/)
  ✗ Data validation (that's processor/)
  ✗ Workflow orchestration (that's core/)
```

#### `src/utils/` — Shared Infrastructure
```
Responsibility: Provide domain-agnostic utilities
Why separate:
  ✓ Reusable across multiple projects
  ✓ No circular dependencies on other project modules
  ✓ Pure functions where possible

What belongs here:
  ✓ Logging configuration
  ✓ Custom exceptions
  ✓ Data augmentation transforms
  ✓ Helper functions (path handling, etc.)
  
What does NOT belong here:
  ✗ Project-specific business logic
  ✗ Configuration values
  ✗ Class implementations with state
```

### 3.2 Class-Level Responsibilities

Each class has a single well-defined contract:

```python
class Holistic86Extractor:
    """Extract landmarks from video using MediaPipe Holistic.
    
    Single Responsibility: 
      Take video path → return numpy array of shape (T, 86, C)
    
    What this class does:
      ✓ Initialize MediaPipe model once (stateful)
      ✓ Process frames through model
      ✓ Aggregate frame results
      
    What this class does NOT do:
      ✗ Decide which keypoints to extract (that's config/)
      ✗ Format or validate keypoints (that's processor/)
      ✗ Save results (that's converter/)
    """
    
    def __init__(self):
        self.model = self._initialize_mediapipe()
    
    def extract_video(self, video_path: str) -> np.ndarray:
        """Single responsibility: video → keypoints."""
        pass
```

### 3.3 Function-Level Responsibilities

Each function performs exactly one logical task:

```python
def extract_frame(self, frame: np.ndarray) -> np.ndarray:
    """Single responsibility: Infer keypoints from one frame.
    
    Does NOT:
      - Load video
      - Process multiple frames
      - Validate output
      - Save results
    """
    pass

def _aggregate_regions(self, results) -> np.ndarray:
    """Single responsibility: Combine body regions into unified array.
    
    Takes MediaPipe output → returns (86, 2) array
    """
    pass
```

---

## 4. Directory Structure and Module Organization

### 4.1 Complete Project Structure

```
rgb-to-skeleton-mediapipe/
├── main.py                          # Entry point - CLI dispatch
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment specification
├── README.md                        # User-facing documentation
├── CODING_STANDARDS.md              # This document
├── LICENSE                          # MIT License
│
├── src/                             # Python package root
│   ├── __init__.py
│   │
│   ├── config/                      # ⭐ Layer 4: Configuration
│   │   ├── __init__.py              # Central re-export point
│   │   ├── settings.py              # MediaPipe, output settings
│   │   ├── keypoint_layout.py       # ⭐ NEW: Keypoint structure & selections
│   │   ├── mappings.py              # ID mapping tables
│   │   └── paths.py                 # Directory structure
│   │
│   ├── core/                        # ⭐ Layer 2: Orchestration
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Main orchestrator
│   │   ├── cli.py                   # Command-line interface
│   │   └── metadata.py              # Video ID parsing
│   │
│   ├── extractor/                   # ⭐ Layer 3a: Feature Extraction
│   │   ├── __init__.py
│   │   └── holistic_86.py           # MediaPipe Holistic wrapper
│   │
│   ├── processor/                   # ⭐ Layer 3b: Data Processing
│   │   ├── __init__.py
│   │   ├── keypoint_selector.py     # ⭐ NEW: Landmark selection & formatting
│   │   └── validators.py            # Data validation logic
│   │
│   ├── converter/                   # ⭐ Layer 3c: Serialization
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract converter interface
│   │   ├── to_pickle.py             # Pickle format
│   │   └── to_excel.py              # Excel format
│   │
│   └── utils/                       # ⭐ Layer 5: Infrastructure
│       ├── __init__.py
│       ├── logger.py                # Logging configuration
│       ├── exceptions.py            # Custom exception classes
│       ├── augmentation.py          # Data augmentation
│       └── helpers.py               # Utility functions
│
├── tests/                           # ⭐ Automated test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_keypoint_layout.py  # Config validation
│   │   ├── test_keypoint_selector.py
│   │   ├── test_holistic_86.py
│   │   ├── test_converters.py
│   │   └── test_metadata.py
│   ├── integration/
│   │   └── test_pipeline_e2e.py
│   └── fixtures/
│       ├── mock_mediapipe.py
│       └── sample_data.py
│
├── data/                            # ⭐ Data directory (in .gitignore)
│   ├── raw/                         # Input video files
│   ├── pickle/                      # Output pickle serialization
│   ├── excel/                       # Output Excel sheets
│   └── results/                     # Processing results
│
└── splitting_data/                  # Data split configuration
    └── results/                     # train.csv, dev.csv, test.csv
```

### 4.2 Module Ownership and Change Frequency

| Module | Primary Purpose | Owner | Update Frequency | Stability |
|--------|-----------------|-------|------------------|-----------|
| `config/keypoint_layout.py` | Define keypoint structure | Researchers | Per study | Moderate (changes = new experiments) |
| `config/settings.py` | MediaPipe parameters | ML Engineer | On model updates | Low |
| `extractor/` | ML integration | ML Engineer | On framework updates | Low |
| `processor/` | Data processing | Data Engineer | Regular | High (frequently extended) |
| `converter/` | Output formats | Data Engineer | On new format requests | High |
| `core/` | Orchestration | Core Team | Regular | Moderate |
| `utils/` | Infrastructure | All | As needed | Stable |

---

## 5. Naming Conventions

Consistent naming is mandatory for long-term maintainability and peer review.

### 5.1 File and Directory Naming

| Element | Convention | Example | Rationale |
|---------|-----------|---------|-----------|
| Module file | `snake_case.py` | `keypoint_selector.py` | PEP 8 standard |
| Package folder | `snake_case/` | `extractor/`, `converter/` | Lowercase, descriptive |
| Class name | `PascalCase` | `Holistic86Extractor`, `KeypointSelector` | Python convention |
| Public function | `snake_case()` | `extract_video()`, `validate_shape()` | Verb-noun pattern |
| Private method | `_snake_case()` | `_aggregate_regions()` | Single underscore prefix |
| Constant | `UPPER_SNAKE_CASE` | `TOTAL_KEYPOINTS`, `PICKLE_DIR` | All caps convention |
| Variable | `snake_case` | `keypoints`, `video_id`, `output_path` | Descriptive, no single letters |
| Test file | `test_<module>.py` | `test_keypoint_selector.py` | pytest convention |
| Config module | `<domain>.py` | `keypoint_layout.py`, `settings.py` | Domain-specific |

### 5.2 Identifier and ID Naming

**Video ID Format** (standardized): `Pxx_Sxxx_Rxx`
- `P`: Person identifier (e.g., `P01`, `P02`)
- `S`: Sentence/Gloss identifier (e.g., `S001`, `S030`)
- `R`: Repetition number (e.g., `R01`, `R05`)

**Variable Naming for IDs**:
```python
video_id: str = "P01_S005_R01"        # Complete ID
person_id: str = "P01"                 # Person component
sentence_id: str = "S005"              # Sentence component
repetition_id: str = "R01"             # Repetition component
```

**Configuration Key Naming**:
```python
# In config files
MOUTH_SELECTION: List[int] = [...]    # What to select
LEFT_HAND_SELECTION: List[int] = [...] # Clear intent
MEDIAPIPE_CONFIG: Dict[str, Any] = {...} # Settings object
```

### 5.3 Docstring Format (Google Style)

```python
def extract_frame(self, frame: np.ndarray) -> np.ndarray:
    """
    Extract 86 keypoints from a single video frame using MediaPipe Holistic.
    
    This method processes an individual BGR frame through the MediaPipe Holistic
    model and returns normalized keypoints in standardized format matching the
    86-point skeleton specification.
    
    Args:
        frame (np.ndarray): Input frame from video capture in BGR color space.
                           Shape must be (height, width, 3). Automatically handles
                           various resolutions; no pre-scaling required.
    
    Returns:
        np.ndarray: Extracted keypoints with shape (86, 2) representing normalized
                   (x, y) coordinates in range [0.0, 1.0]. Does not include z-axis
                   to reduce data dimensionality per project specification.
    
    Raises:
        ValueError: If frame is None, empty, or cannot be converted to RGB.
        
    Notes:
        - All coordinates are automatically normalized by MediaPipe
        - Missing detections (e.g., occluded hand) return zero vectors
        - Single frame processing is stateless (no internal state modification)
        
    Example:
        >>> import cv2
        >>> from src.extractor.holistic_86 import Holistic86Extractor
        >>> frame = cv2.imread("image.jpg")
        >>> extractor = Holistic86Extractor()
        >>> keypoints = extractor.extract_frame(frame)
        >>> print(keypoints.shape)
        (86, 2)
        >>> print(keypoints[0, :])  # First keypoint (left hand landmark 0)
        array([0.512, 0.341])
    """
```

---

## 6. Core Architecture Patterns

### 6.1 Configuration-Driven Architecture

The system separates **configuration** (WHAT) from **implementation** (HOW):

#### Pattern: Keypoint Layout Configuration

**File**: `src/config/keypoint_layout.py`

```python
"""
Keypoint Layout Configuration Module

Defines the complete 86-keypoint structure using MediaPipe Holistic.
All landmark selections are centralized for reproducibility and easy modification.

This module embodies the principle: "Configuration defines what the system 
extracts, not the code logic."
"""

from typing import Dict, List, Tuple

# Region definitions with index ranges
KEYPOINT_RANGES: Dict[str, Tuple[int, int]] = {
    "left_hand": (0, 21),      # Keypoints 0–20 (21 total)
    "right_hand": (21, 42),    # Keypoints 21–41 (21 total)
    "mouth": (42, 61),         # Keypoints 42–60 (19 total)
    "pose": (61, 86),          # Keypoints 61–85 (25 total)
}

# Landmark selections (indices from MediaPipe output)
LEFT_HAND_SELECTION: List[int] = list(range(0, 21))   # Sequential
RIGHT_HAND_SELECTION: List[int] = list(range(0, 21))  # Sequential

# Mouth: Custom selection from 468-point face mesh
# Only lip region, not eye/nose area
MOUTH_SELECTION: List[int] = [
    # Outer lip (10 points clockwise)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Inner lip (9 points)
    78, 191, 80, 81, 82, 13, 312, 311, 308,
]

POSE_SELECTION: List[int] = list(range(0, 25))  # Top 25 poses

# Aggregated lookup
KEYPOINT_SELECTIONS: Dict[str, List[int]] = {
    "left_hand": LEFT_HAND_SELECTION,
    "right_hand": RIGHT_HAND_SELECTION,
    "mouth": MOUTH_SELECTION,
    "pose": POSE_SELECTION,
}
```

**Key Benefits**:
- ✅ Researchers can modify keypoint selections without touching extraction code
- ✅ Different studies can use different configurations from the same codebase
- ✅ Configuration changes are explicitly tracked in version control
- ✅ Easy to document which keypoints were used for reproducibility

#### Pattern: Keypoint Selection Processor

**File**: `src/processor/keypoint_selector.py`

```python
class KeypointSelector:
    """
    Selects and formats landmarks from MediaPipe Holistic output.
    
    Single Responsibility: Transform raw MediaPipe landmarks into structured
    numpy arrays using configured selection indices.
    
    Decouples:
      - Configuration (what to select) → config/keypoint_layout.py
      - Selection algorithm (how to select) → this class
      - Usage (where to select) → extractor uses this class
    """
    
    def __init__(self, use_3d: bool = False):
        self.use_3d = use_3d
        self.dims = 3 if use_3d else 2
    
    def select_landmarks(
        self,
        landmarks,
        selection: List[int],
    ) -> List[List[float]]:
        """
        Select specific landmarks from MediaPipe output.
        
        Parameters:
            landmarks: MediaPipe landmarks object (or None)
            selection: List of indices to extract (e.g., [0, 1, 2, ...])
        
        Returns:
            List of coordinate lists (validated format)
        
        This method is model-agnostic: works with any landmark source
        as long as it follows MediaPipe's landmark protocol.
        """
        if landmarks is None:
            return [[0.0] * self.dims for _ in selection]
        
        output = []
        for idx in selection:
            lm = landmarks.landmark[idx]
            coords = [lm.x, lm.y]
            if self.use_3d:
                coords.append(lm.z)
            output.append(coords)
        
        return output
```

### 6.2 Extractor with Configuration Integration

**File**: `src/extractor/holistic_86.py`

```python
from src.config.keypoint_layout import (
    LEFT_HAND_SELECTION,
    RIGHT_HAND_SELECTION,
    MOUTH_SELECTION,
    POSE_SELECTION,
)
from src.processor.keypoint_selector import KeypointSelector

class Holistic86Extractor:
    """Extract 86 keypoints using MediaPipe Holistic.
    
    Architecture:
      1. Initialize MediaPipe model (once, at __init__)
      2. For each frame:
         a. Process through model
         b. Use KeypointSelector to extract configured landmarks
         c. Aggregate into structured array
      3. Return video-level output
    
    Does NOT include:
      - Hardcoded landmark indices
      - Keypoint validation logic
      - Output serialization
    """
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(**MEDIAPIPE_CONFIG)
        self.selector = KeypointSelector(use_3d=USE_3D_COORDINATES)
    
    def _aggregate_regions(self, results) -> np.ndarray:
        """Combine body regions using configured selections."""
        keypoints = []
        
        # Use selections from config (not hardcoded)
        keypoints.extend(
            self.selector.select_landmarks(
                results.left_hand_landmarks,
                LEFT_HAND_SELECTION,
            )
        )
        keypoints.extend(
            self.selector.select_landmarks(
                results.right_hand_landmarks,
                RIGHT_HAND_SELECTION,
            )
        )
        keypoints.extend(
            self.selector.select_landmarks(
                results.face_landmarks,
                MOUTH_SELECTION,
            )
        )
        keypoints.extend(
            self.selector.select_landmarks(
                results.pose_landmarks,
                POSE_SELECTION,
            )
        )
        
        return np.array(keypoints)
    
    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from one frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)
        return self._aggregate_regions(results)
    
    def extract_video(self, video_path: str) -> np.ndarray:
        """Extract keypoints from all frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ExtractionException(f"Cannot open video: {video_path}")
        
        try:
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(self.extract_frame(frame))
            
            if not all_frames:
                raise ExtractionException(f"No frames in video: {video_path}")
            
            return np.stack(all_frames)  # Shape: (T, 86, 2)
        
        finally:
            cap.release()
```

### 6.3 Converter Strategy Pattern

```python
from abc import ABC, abstractmethod

class BaseConverter(ABC):
    """Abstract base class for all output format converters."""
    
    @abstractmethod
    def save(
        self,
        keypoints: np.ndarray,
        video_id: str,
        output_subpath: str = "",
    ) -> Tuple[str, str]:
        """Save keypoints in target format.
        
        Returns:
            Tuple of (sample_id, file_path)
        """
        pass

class PickleConverter(BaseConverter):
    """Serialize keypoints to Pickle format."""
    
    def save(self, keypoints, video_id, output_subpath=""):
        # Pickle-specific logic
        pass

class ExcelConverter(BaseConverter):
    """Export keypoints to Excel tabular format."""
    
    def save(self, keypoints, video_id, output_subpath=""):
        # Excel-specific logic
        pass

# Adding new format requires only:
# 1. Create new_converter.py with class implementing BaseConverter
# 2. Register in pipeline
# 3. NO changes to existing code (Open/Closed Principle)
```

### 6.4 Pipeline Orchestration

```python
class SkeletonPipeline:
    """Orchestrate the complete extraction → processing → serialization workflow."""
    
    def __init__(self, converters: List[BaseConverter] = None):
        self.extractor = Holistic86Extractor()
        self.converters = converters or [
            PickleConverter(),
            ExcelConverter(),
        ]
        self.split_mapping = self._load_split_mapping()
    
    def process_video(self, video_path: str) -> np.ndarray:
        """Process single video through complete pipeline."""
        
        # Extract
        keypoints = self.extractor.extract_video(video_path)
        
        # Parse metadata
        video_id = parse_video_id(Path(video_path))
        
        # Route and save
        for converter in self.converters:
            converter.save(keypoints, video_id)
        
        return keypoints
    
    def process_folder(self, folder_path: str) -> None:
        """Batch process all videos in folder."""
        videos = Path(folder_path).rglob("*.mp4")
        
        for video_path in sorted(videos):
            try:
                self.process_video(str(video_path))
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
```

---

## 7. Python Code Standards

### 7.1 Type Annotations (Mandatory)

All public functions must have complete type hints:

```python
# ✅ CORRECT
def extract_video(self, video_path: str) -> np.ndarray:
    pass

def get_output_path(
    self,
    video_id: str,
    output_subpath: str = "",
) -> Path:
    pass

# ❌ INCORRECT - No type hints
def extract_video(self, video_path):
    pass
```

**Type Annotation Guidelines**:
- Use `| None` instead of `Optional` (Python 3.10+)
- Use `Dict[K, V]`, `List[T]`, `Tuple[T, ...]` from typing module
- Never use bare `Any` without justification and comment
- For numpy: `np.ndarray` (with shape documentation in docstring)

### 7.2 Code Style (PEP 8 + Black)

```

**Format specifications**:
- Line length: 100 characters maximum (Black default)
- Indentation: 4 spaces (never tabs)
- Blank lines: 2 before module-level definitions, 1 before method definitions
- Import order: stdlib → 3rd party → local (all alphabetical within groups)

```python
# Correct import order
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import MEDIAPIPE_CONFIG
from src.extractor.holistic_86 import Holistic86Extractor
```

### 7.3 Naming Best Practices

```python
# ✅ GOOD - Clear intent
model_complexity = 2
min_detection_confidence = 0.5
extract_mouth_landmarks = True

# ❌ AVOID - Ambiguous or unclear
mc = 2
min_conf = 0.5
extract_mouth = True
```

---

## 8. Data Structure Specifications

### 8.1 Keypoint Array Format

**Standard output shape**: `(T, 86, 2)`

```python
# T = number of frames (variable per video)
# 86 = fixed number of keypoints
# 2 = coordinate dimensions (x, y only, no z-axis)

# Example:
keypoints = np.random.rand(250, 86, 2)  # 250 frames

# Coordinate ranges:
assert keypoints.min() >= 0.0
assert keypoints.max() <= 1.0  # Normalized coordinates

# Structure:
# keypoints[frame_idx, keypoint_idx, coord_idx]
#
# Example access:
left_hand_first_kpt = keypoints[0, 0, :]      # Shape (2,)
all_mouth_frame_0 = keypoints[0, 42:61, :]    # Shape (19, 2)
```

### 8.2 Pickle Dictionary Format

**Standard format** (produced by PickleConverter):

```python
{
    "P01_S001_R01": {
        "keypoints": np.ndarray,  # Shape (T, 86, 2), dtype float64
    },
    "P01_S001_R02": {
        "keypoints": np.ndarray,
    },
    # ... more samples ...
}
```

**Validation**:
```python
def validate_pickle_dict(data: Dict[str, Dict]) -> bool:
    """Validate pickle dictionary structure."""
    assert isinstance(data, dict), "Root must be dictionary"
    
    for video_id, entry in data.items():
        assert isinstance(video_id, str), f"Video ID must be string: {video_id}"
        assert video_id.count('_') == 2, f"Invalid ID format: {video_id}"
        
        keypoints = entry["keypoints"]
        assert keypoints.ndim == 3, f"Wrong shape: {keypoints.shape}"
        assert keypoints.shape[1] == 86, f"Wrong keypoint count: {keypoints.shape}"
        assert keypoints.shape[2] == 2, f"Wrong coordinate dimension: {keypoints.shape}"
        assert keypoints.dtype == np.float64, f"Wrong dtype: {keypoints.dtype}"
    
    return True
```

---

## 9. Configuration Management Framework

### 9.1 Configuration Hierarchy

Configuration is loaded in this priority order:

```
Level 1: Environment Variables (highest priority)
  Source: OS environment at runtime
  Usage: MEDIAPIPE_COMPLEXITY=2 python main.py

Level 2: Configuration Files (default values)
  Source: src/config/*.py
  Usage: MEDIAPIPE_CONFIG from settings.py

Level 3: Runtime Arguments (CLI flags)
  Source: Command-line arguments
  Usage: python main.py --input data/raw/
```

### 9.2 Configuration Module Structure

```python
# src/config/__init__.py - Central re-export point
from .settings import MEDIAPIPE_CONFIG, USE_3D_COORDINATES
from .keypoint_layout import KEYPOINT_SELECTIONS, TOTAL_KEYPOINTS
from .mappings import PERSON_MAP, SENTENCE_MAP
from .paths import PICKLE_DIR, EXCEL_DIR, RAW_VIDEO_DIR

# External code imports from single point:
from src.config import MEDIAPIPE_CONFIG, PICKLE_DIR
# NOT:
from src.config.settings import MEDIAPIPE_CONFIG
from src.config.paths import PICKLE_DIR
```

---

## 10. Error Handling and Logging

### 10.1 Custom Exception Hierarchy

```python
# src/utils/exceptions.py

class SkeletonPipelineException(Exception):
    """Base exception for all pipeline errors."""
    pass

class ExtractionException(SkeletonPipelineException):
    """Raised when video extraction fails (codec, corruption, etc.)."""
    pass

class ValidationException(SkeletonPipelineException):
    """Raised when data validation fails (shape, bounds, etc.)."""
    pass

class ConfigurationException(SkeletonPipelineException):
    """Raised when configuration is invalid."""
    pass

class ConversionException(SkeletonPipelineException):
    """Raised when output format conversion fails."""
    pass
```

### 10.2 Logging Configuration

```python
# src/utils/logger.py
import logging

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get configured logger for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Usage in modules:
logger = get_logger(__name__)
logger.info(f"Processing video: {video_path}")
```

---

## 11. Testing and Validation Strategy

### 11.1 Test Structure and Pyramid

```
Tests (80% unit, 15% integration, 5% E2E)

Unit Tests (80%)
  ├── test_keypoint_layout.py      # Config validation
  ├── test_keypoint_selector.py    # Selection logic
  ├── test_holistic_86.py          # Extraction
  ├── test_converters.py           # Serialization
  └── test_metadata.py             # Parsing

Integration Tests (15%)
  ├── test_pipeline_e2e.py         # Full workflow
  └── test_converter_pipeline.py   # Multi-format output

E2E Tests (5%)
  └── test_real_world_video.py     # Sample dataset
```

### 11.2 Unit Test Requirements

Every public class and function must have corresponding unit tests:

```python
# tests/unit/test_holistic_86.py
import unittest
import numpy as np
from src.extractor.holistic_86 import Holistic86Extractor
from src.utils.exceptions import ExtractionException

class TestHolistic86Extractor(unittest.TestCase):
    """Test suite for Holistic86Extractor."""
    
    def setUp(self):
        """Initialize extractor for each test."""
        self.extractor = Holistic86Extractor()
    
    def test_extract_frame_returns_correct_shape(self):
        """Verify frame extraction produces (86, 2) output."""
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        keypoints = self.extractor.extract_frame(frame)
        
        self.assertEqual(keypoints.shape, (86, 2))
        self.assertEqual(keypoints.dtype, np.float64)
    
    def test_extract_frame_coordinates_normalized(self):
        """Verify coordinates are in [0, 1] range."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        keypoints = self.extractor.extract_frame(frame)
        
        self.assertGreaterEqual(keypoints.min(), 0.0)
        self.assertLessEqual(keypoints.max(), 1.0)
    
    def test_extract_video_raises_on_missing_file(self):
        """Verify appropriate error for missing video."""
        with self.assertRaises(ExtractionException):
            self.extractor.extract_video("nonexistent.mp4")
    
    def tearDown(self):
        """Clean up after tests."""
        pass
```

### 11.3 Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_holistic_86.py -v

# Run tests matching pattern
pytest tests/ -k "test_extraction" -v
```

**Coverage targets**:
- Critical modules (extractor, processor): **90%+**
- Converters, validators: **85%+**
- Configuration validation: **80%+**
- Overall project: **80%+**

---

## 12. SOLID Design Principles

This project strictly adheres to SOLID principles:

### 12.1 Single Responsibility (SRP)
- ✅ Each module has exactly one reason to change
- ✅ Each class has single, well-defined contract
- ✅ Each function performs one logical task

### 12.2 Open/Closed (OCP)
- ✅ Pipeline open for extension (new converters, extractors)
- ✅ Core logic closed for modification
- Example: Add new converter without touching pipeline code

### 12.3 Liskov Substitution (LSP)
- ✅ All converters implement BaseConverter interface
- ✅ Different converters can be swapped without changing pipeline
- ✅ Consistent behavior contract across all implementations

### 12.4 Interface Segregation (ISP)
- ✅ Interfaces are focused and minimal
- ✅ Converters only implement `save()` method, nothing extra
- ✅ No "fat" interfaces with unnecessary methods

### 12.5 Dependency Inversion (DIP)
- ✅ Pipeline depends on abstractions (BaseConverter), not concrete classes
- ✅ Configuration is dependency-injected, not hardcoded
- ✅ Easy to mock for testing

---

## 13. Scalability and Maintainability

### 13.1 Adding New Extractor

To integrate a different pose estimation model (e.g., OpenPose):

```python
# src/extractor/openpose_25.py
class OpenPose25Extractor:
    """Extract 25 keypoints using OpenPose model."""
    
    def extract_video(self, video_path: str) -> np.ndarray:
        """Same interface as Holistic86Extractor."""
        # OpenPose-specific implementation
        return np.ndarray  # Shape: (T, 25, 2)

# In pipeline:
from src.extractor.openpose_25 import OpenPose25Extractor

pipeline = SkeletonPipeline(
    extractor=OpenPose25Extractor()  # Just inject different extractor
)
```

**No changes needed to**: core logic, converters, or configuration.

### 13.2 Adding New Output Format

To export to HDF5 instead of Pickle:

```python
# src/converter/to_hdf5.py
import h5py

class HDF5Converter(BaseConverter):
    """Export keypoints to HDF5 format."""
    
    def save(self, keypoints, video_id, output_subpath=""):
        output_path = self.get_output_path(video_id, output_subpath)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(video_id, data=keypoints)
        
        return video_id, output_path

# In pipeline:
pipeline = SkeletonPipeline(
    converters=[
        PickleConverter(),
        ExcelConverter(),
        HDF5Converter(),  # Just add new converter
    ]
)
```

**No changes needed to**: extraction, processing, or core pipeline.

---

## 14. Anti-Patterns and What to Avoid

### 14.1 Hardcoded Magic Numbers

```python
# ❌ ANTI-PATTERN
if keypoints.shape[1] != 86:  # What's 86?
    raise ValueError("Invalid keypoint count")

# ✅ CORRECT
from src.config.keypoint_layout import TOTAL_KEYPOINTS

if keypoints.shape[1] != TOTAL_KEYPOINTS:
    raise ValueError(f"Invalid keypoint count: expected {TOTAL_KEYPOINTS}")
```

### 14.2 Scattered Configuration

```python
# ❌ ANTI-PATTERN
# config 1: in pipeline.py
MEDIAPIPE_CONFIG = {"complexity": 2}

# config 2: in extractor.py
MIN_CONFIDENCE = 0.5

# config 3: in converter.py
PICKLE_FILENAME = "pose_bisindo"

# ✅ CORRECT
# All in src/config/*.py
from src.config import MEDIAPIPE_CONFIG, MIN_CONFIDENCE, PICKLE_FILENAME
```

### 14.3 Deep Inheritance

```python
# ❌ ANTI-PATTERN
class BaseProcessor:
    pass

class VideoExtractor(BaseProcessor):
    pass

class Holistic86Extractor(VideoExtractor):
    pass

# ✅ CORRECT
class Holistic86Extractor:  # Direct implementation
    pass
```

### 14.4 Mixed Responsibilities

```python
# ❌ ANTI-PATTERN
class VideoProcessor:
    def extract_keypoints(self):
        pass
    
    def validate_keypoints(self):
        pass
    
    def save_to_pickle(self):
        pass
    
    def save_to_excel(self):
        pass

# ✅ CORRECT
class Holistic86Extractor:
    def extract_keypoints(self):
        pass

class KeypointSelector:
    def validate_keypoints(self):
        pass

class PickleConverter:
    def save(self):
        pass

class ExcelConverter:
    def save(self):
        pass
```

---

## 15. Research Paper Reproducibility

For Q1 research publication, ensure reproducibility:

### 15.1 Configuration Versioning

```python
# src/config/__init__.py
PROJECT_VERSION = "2.0.0"
MEDIAPIPE_VERSION = "0.10.14"

# Always document what versions were used
CONFIGURATION_METADATA = {
    "project_version": PROJECT_VERSION,
    "mediapipe_version": MEDIAPIPE_VERSION,
    "mediapipe_model_complexity": 2,
    "keypoint_count": 86,
    "coordinate_dimensions": 2,
    "keypoint_layout": "Isharah Format",
    "extraction_date": "2026-05-14",
}
```

### 15.2 Output Metadata

```python
# Save configuration with outputs
def save_with_metadata(
    keypoints: np.ndarray,
    video_id: str,
    config_metadata: Dict,
):
    """Save keypoints with full configuration metadata."""
    pickle_data = {
        video_id: {
            "keypoints": keypoints,
            "metadata": {
                "extraction_config": config_metadata,
                "extraction_timestamp": datetime.now().isoformat(),
            }
        }
    }
    # ... save to file ...
```

### 15.3 Code Freezing for Publication

Before submitting to conference:

```bash
# 1. Tag exact version used
git tag -a v2.0.0-publication -m "Version used for Q1 submission"

# 2. Document configuration
python -c "from src.config import CONFIGURATION_METADATA; print(CONFIGURATION_METADATA)"

# 3. Generate reproducibility report
python scripts/generate_repro_report.py > reproducibility_report.txt

# 4. Archive source code
git archive --format=zip HEAD > rgb-skeleton-v2.0.0-publication.zip
```

---

## 16. Contributor Guidelines

### 16.1 Pre-Contribution Checklist

Before submitting a pull request:

- [ ] Read and understood CODING_STANDARDS.md (this document)
- [ ] Code follows naming conventions (PascalCase for classes, snake_case for functions)
- [ ] All functions have complete type hints
- [ ] All public functions have Google-style docstrings
- [ ] No hardcoded values (use constants from config/)
- [ ] Unit tests written for new code
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code formatted: `black src/`
- [ ] No linting errors: `pylint src/`
- [ ] Type checking passes: `mypy src/`
- [ ] Coverage acceptable: `pytest --cov=src`
- [ ] Commit message is descriptive: `feat: add HDF5 converter`

### 16.2 Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-converter

# 2. Write code following standards
# ... edit files ...

# 3. Write tests
# ... create tests/unit/test_new_feature.py ...

# 4. Run quality checks
black src/
pylint src/
mypy src/
pytest tests/ --cov=src

# 5. Commit with clear message
git commit -m "feat: add HDF5 converter

- Implements BaseConverter interface
- Supports append mode for batch processing
- Includes full test coverage"

# 6. Push and create PR
git push origin feature/new-converter
```

---

## 17. Quality Assurance Checklist

### Code Quality Metrics

```
Cyclomatic Complexity:  ≤ 10 per function
Function Length:        ≤ 50 lines ideal, ≤ 100 max
Class Size:            ≤ 300 lines ideal, ≤ 500 max
Type Hint Coverage:    100%
Docstring Coverage:    100% of public APIs
Test Coverage:         ≥ 80% overall
                       ≥ 90% for critical modules
Line Length:           ≤ 100 characters
```

### Automated Quality Checks

```bash
# Run all quality checks
bash scripts/quality-check.sh

# Individual commands:
black src/ --check
pylint src/ --exit-zero
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Code Review Template

```markdown
## Code Review Checklist

### Architecture
- [ ] Follows Single Responsibility Principle
- [ ] No circular dependencies
- [ ] Appropriate separation of concerns
- [ ] SOLID principles applied

### Implementation
- [ ] All functions have type hints
- [ ] Docstrings complete and accurate
- [ ] Constants centralized in config/
- [ ] No magic numbers

### Testing
- [ ] Unit tests comprehensive
- [ ] Edge cases covered
- [ ] Tests are independent
- [ ] Coverage ≥ 80%

### Performance
- [ ] No obvious algorithmic issues
- [ ] Appropriate data structures
- [ ] Model instance reused (not recreated per frame)

### Documentation
- [ ] README updated if needed
- [ ] CODING_STANDARDS updated if relevant
- [ ] Comments explain "why", not "what"
```

---

## Summary: Key Takeaways

This project implements **strict Single Responsibility Principle** at package, class, and function levels. Key principles:

1. **Configuration Defines WHAT** (src/config/)
2. **Implementation Defines HOW** (src/extractor/, processor/, converter/)
3. **Pipeline Defines WHERE** (src/core/)
4. **Type Safety is Mandatory** (all functions typed)
5. **Testing is Non-Negotiable** (80%+ coverage required)
6. **Reproducibility for Research** (configuration tracked, versions frozen)

By following these standards, the codebase remains maintainable, scalable, and suitable for Q1 research publication.

---

**Document Version**: 2.0.0  
**Last Updated**: May 14, 2026  
**Status**: Approved for Q1 Research Publication  
**Next Review**: Upon significant architecture changes
