import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

from app.services.detector import Detection, YOLODetector
from app.utils.visualizer import draw_detections, draw_legend
from app.config import get_settings

settings = get_settings()


@dataclass
class VideoMetadata:
    """Video metadata container."""
    
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    size_mb: float
    codec: str = ""


@dataclass
class ProcessingStats:
    """Processing statistics container."""
    
    processing_time: float = 0.0
    frames_processed: int = 0
    total_detections: int = 0
    players_count: int = 0
    goalkeepers_count: int = 0
    referees_count: int = 0
    balls_count: int = 0
    detections_per_frame: List[int] = field(default_factory=list)


class VideoProcessor:
    """Video processing service."""
    
    def __init__(self, detector: YOLODetector = None):
        """Initialize video processor.
        
        Args:
            detector: YOLO detector instance
        """
        self.detector = detector
        
        # Ensure temp directory exists
        os.makedirs(settings.temp_dir, exist_ok=True)
    
    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        cap.release()
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file size
        size_bytes = os.path.getsize(video_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Decode codec
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return VideoMetadata(
            filename=os.path.basename(video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            size_mb=size_mb,
            codec=codec
        )
    
    def extract_frames(
        self,
        video_path: str,
        skip_frames: int = 0
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract frames from video.
        
        Args:
            video_path: Path to video file
            skip_frames: Number of frames to skip between processed frames
            
        Yields:
            Tuple of (frame_index, frame)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if skip_frames == 0 or frame_idx % (skip_frames + 1) == 0:
                yield frame_idx, frame
                
            frame_idx += 1
            
        cap.release()
    
    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        draw_legend_flag: bool = True
    ) -> Tuple[str, VideoMetadata, ProcessingStats]:
        """Process video with object detection.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (optional, auto-generated if None)
            draw_legend_flag: Whether to draw class legend
            
        Returns:
            Tuple of (output_path, metadata, stats)
        """
        if self.detector is None:
            raise ValueError("Detector not initialized")
        
        # Get video metadata
        metadata = self.get_metadata(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(
                settings.temp_dir,
                f"output_{int(time.time())}_{metadata.filename}"
            )
            # Ensure .mp4 extension
            if not output_path.endswith('.mp4'):
                output_path = os.path.splitext(output_path)[0] + '.mp4'
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height)
        )
        
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        
        # Processing stats
        stats = ProcessingStats()
        start_time = time.time()
        
        try:
            for frame_idx, frame in self.extract_frames(input_path):
                # Run detection
                detections = self.detector.detect(frame)
                
                # Update stats
                stats.frames_processed += 1
                stats.detections_per_frame.append(len(detections))
                stats.total_detections += len(detections)
                
                for det in detections:
                    if det.class_id == 0:
                        stats.players_count += 1
                    elif det.class_id == 1:
                        stats.goalkeepers_count += 1
                    elif det.class_id == 2:
                        stats.referees_count += 1
                    elif det.class_id == 3:
                        stats.balls_count += 1
                
                # Draw detections
                annotated = draw_detections(frame, detections)
                
                # Draw legend on first few frames and periodically
                if draw_legend_flag and frame_idx % 1 == 0:
                    annotated = draw_legend(annotated)
                
                # Write frame
                writer.write(annotated)
                
        finally:
            writer.release()
            
        stats.processing_time = time.time() - start_time
        
        return output_path, metadata, stats
    
    def process_video_to_bytes(
        self,
        input_path: str,
        draw_legend_flag: bool = True
    ) -> Tuple[bytes, VideoMetadata, ProcessingStats]:
        """Process video and return as bytes.
        
        Args:
            input_path: Path to input video
            draw_legend_flag: Whether to draw class legend
            
        Returns:
            Tuple of (video_bytes, metadata, stats)
        """
        # Process to temp file
        output_path = os.path.join(
            settings.temp_dir,
            f"temp_output_{int(time.time())}.mp4"
        )
        
        try:
            output_path, metadata, stats = self.process_video(
                input_path,
                output_path,
                draw_legend_flag
            )
            
            # Read output file as bytes
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                
            return video_bytes, metadata, stats
            
        finally:
            # Cleanup temp output file
            if os.path.exists(output_path):
                os.remove(output_path)


def get_video_processor(detector: YOLODetector = None) -> VideoProcessor:
    """Create video processor instance.
    
    Args:
        detector: Optional detector instance
        
    Returns:
        VideoProcessor instance
    """
    return VideoProcessor(detector)

