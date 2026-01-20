"""Video Scoring Module - Real-time video quality analysis with qwen-agent framework

Input: Video file path
Output: Video score (0-100)

This module uses qwen-agent framework to analyze video quality across multiple dimensions.
Supports real video analysis: ffprobe for technical metrics, OpenCV for visual quality, Qwen for content.
"""

import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict
from http import HTTPStatus

import dashscope

if not getattr(dashscope, "api_key", "") or not str(getattr(dashscope, "api_key", "")).strip():
    try:
        from config_loader import get_dashscope_api_key
        dashscope.api_key = get_dashscope_api_key()
    except (ImportError, FileNotFoundError, ValueError, KeyError):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@register_tool("video_analyzer")
class VideoAnalyzer(BaseTool):
    """Video analysis tool - analyzes video quality across multiple dimensions"""

    description = (
        "Analyzes video quality from multiple dimensions including: "
        "visual quality, audio quality, content coherence, motion smoothness, "
        "and overall engagement. Returns detailed scores and recommendations."
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file to be analyzed"
            },
            "analysis_type": {
                "type": "string",
                "enum": ["visual", "audio", "content", "motion", "overall"],
                "description": "Type of analysis to perform"
            }
        },
        "required": ["video_path"]
    }

    def call(self, video_path: str, analysis_type: str = "overall", **kwargs) -> str:
        """Execute video analysis
        
        Args:
            video_path: Path to video file
            analysis_type: Type of analysis (visual/audio/content/motion/overall)
        
        Returns:
            JSON format analysis results
        """
        if not os.path.exists(video_path):
            return json.dumps({
                "error": f"Video file not found: {video_path}",
                "score": 0
            }, ensure_ascii=False)
        
        # Get video information
        video_info = self._get_video_info(video_path)
        
        # Perform analysis
        if analysis_type == "visual":
            scores = self._analyze_visual_quality(video_path, video_info)
        elif analysis_type == "audio":
            # Audio analysis has been disabled (commented out).
            # The original implementation is retained in the file but will not be executed.
            scores = {
                "analysis_type": "audio",
                "overall_audio_score": None,
                "note": "Audio analysis is disabled (commented out)."
            }
        elif analysis_type == "content":
            scores = self._analyze_content(video_path, video_info)
        elif analysis_type == "motion":
            scores = self._analyze_motion(video_path, video_info)
        else:  # overall
            scores = self._analyze_overall(video_path, video_info)
        
        return json.dumps(scores, ensure_ascii=False)

    def _get_video_info(self, video_path: str) -> dict:
        """Get basic video information"""
        info = {
            "file_size": os.path.getsize(video_path),
            "format": Path(video_path).suffix.lower()
        }
        
        # Use ffprobe to get video metadata
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams", 
                 "-of", "json", video_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                ffprobe_info = json.loads(result.stdout)
                info.update(ffprobe_info)
                return info
        except Exception as e:
            print(f"[INFO] ffprobe analysis failed: {e}")
        
        return info

    def _extract_video_metrics(self, info: dict) -> Dict[str, any]:
        """Extract key metrics from video information"""
        metrics = {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "duration": 10,
            "bitrate": 5000,
            "codec": "h264"
        }
        
        try:
            # Extract from streams
            if "streams" in info:
                for stream in info["streams"]:
                    if stream.get("codec_type") == "video":
                        metrics["width"] = stream.get("width", 1920)
                        metrics["height"] = stream.get("height", 1080)
                        metrics["fps"] = float(stream.get("r_frame_rate", "30/1").split("/")[0])
                        metrics["codec"] = stream.get("codec_name", "h264")
                    elif stream.get("codec_type") == "audio":
                        metrics["audio_codec"] = stream.get("codec_name", "aac")
                        metrics["sample_rate"] = stream.get("sample_rate", 48000)
            
            # Extract duration and bitrate from format
            if "format" in info:
                metrics["duration"] = float(info["format"].get("duration", 10))
                bitrate_str = info["format"].get("bit_rate", "5000000")
                metrics["bitrate"] = int(bitrate_str) / 1000  # Convert to kbps
        except Exception as e:
            print(f"[INFO] Metric extraction failed: {e}")
        
        return metrics

    def _analyze_visual_quality(self, video_path: str, info: dict) -> dict:
        """Analyze visual quality - based on real video metrics"""
        metrics = self._extract_video_metrics(info)
        
        # Resolution score
        width, height = metrics["width"], metrics["height"]
        pixel_count = width * height
        if pixel_count >= 3840 * 2160:  # 4K
            resolution_score = 95
        elif pixel_count >= 1920 * 1080:  # Full HD
            resolution_score = 85
        elif pixel_count >= 1280 * 720:  # HD
            resolution_score = 75
        else:
            resolution_score = 60
        
        # Bitrate score
        bitrate = metrics["bitrate"]
        if bitrate >= 8000:  # High quality
            bitrate_score = 90
        elif bitrate >= 4000:  # Medium quality
            bitrate_score = 80
        elif bitrate >= 1000:
            bitrate_score = 70
        else:
            bitrate_score = 50
        
        # Codec score
        codec = metrics["codec"].lower()
        if "h265" in codec or "hevc" in codec:
            codec_score = 85
        elif "h264" in codec or "avc" in codec:
            codec_score = 80
        else:
            codec_score = 70
        
        # Analyze brightness and contrast with OpenCV
        clarity_score = 75
        brightness_score = 75
        
        if HAS_OPENCV:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                brightness_sum = 0
                contrast_sum = 0
                
                # Analyze first 20 frames
                while frame_count < 20:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate brightness
                    brightness = cv2.mean(gray)[0]
                    brightness_sum += brightness
                    
                    # Calculate contrast (standard deviation)
                    contrast = gray.std()
                    contrast_sum += contrast
                    
                    frame_count += 1
                
                cap.release()
                
                if frame_count > 0:
                    avg_brightness = brightness_sum / frame_count
                    avg_contrast = contrast_sum / frame_count
                    
                    # Brightness score (optimal range: 80-150)
                    if 80 <= avg_brightness <= 150:
                        brightness_score = 85
                    elif 60 <= avg_brightness <= 180:
                        brightness_score = 75
                    else:
                        brightness_score = 60
                    
                    # Clarity score
                    clarity_score = min(90, 40 + avg_contrast / 2)
            except Exception as e:
                print(f"[INFO] OpenCV analysis failed: {e}")
        
        visual_score = (resolution_score * 0.35 + bitrate_score * 0.25 + 
                       codec_score * 0.20 + clarity_score * 0.10 + brightness_score * 0.10)
        
        return {
            "analysis_type": "visual",
            "metrics": {
                "resolution": f"{width}x{height}",
                "bitrate_kbps": round(bitrate, 0),
                "codec": codec
            },
            "scores": {
                "resolution": round(resolution_score, 1),
                "bitrate": round(bitrate_score, 1),
                "codec": round(codec_score, 1),
                "clarity": round(clarity_score, 1),
                "brightness": round(brightness_score, 1)
            },
            "overall_visual_score": round(visual_score, 2),
            "recommendations": self._get_visual_recommendations(visual_score)
        }

    def _analyze_audio_quality(self, video_path: str, info: dict) -> dict:
        """Audio analysis is currently disabled.

        The original implementation (kept here for reference) is commented out below.
        To re-enable audio analysis, restore the original code and remove this placeholder.
        """

        # Placeholder response indicating audio analysis disabled
        return {
            "analysis_type": "audio",
            "metrics": {
                "sample_rate": "N/A",
                "codec": "N/A",
                "has_audio": False
            },
            "scores": {},
            "overall_audio_score": None,
            "recommendations": ["Audio analysis disabled"]
        }

        """
        metrics = self._extract_video_metrics(info)
        
        # Sample rate score
        sample_rate = metrics.get("sample_rate", 48000)
        if sample_rate >= 48000:
            sample_score = 90
        elif sample_rate >= 44100:
            sample_score = 85
        elif sample_rate >= 32000:
            sample_score = 75
        else:
            sample_score = 60
        
        # Audio codec score
        audio_codec = metrics.get("audio_codec", "aac").lower()
        if "aac" in audio_codec:
            codec_score = 85
        elif "mp3" in audio_codec:
            codec_score = 75
        elif "flac" in audio_codec or "pcm" in audio_codec:
            codec_score = 95
        else:
            codec_score = 70
        
        # Check if audio stream exists
        has_audio = False
        if "streams" in info:
            for stream in info["streams"]:
                if stream.get("codec_type") == "audio":
                    has_audio = True
                    break
        
        audio_score = (sample_score * 0.4 + codec_score * 0.4 + 
                      (85 if has_audio else 20) * 0.2)
        
        return {
            "analysis_type": "audio",
            "metrics": {
                "sample_rate": f"{sample_rate}Hz",
                "codec": audio_codec,
                "has_audio": has_audio
            },
            "scores": {
                "sample_rate": round(sample_score, 1),
                "codec": round(codec_score, 1),
                "presence": round(85 if has_audio else 20, 1)
            },
            "overall_audio_score": round(audio_score, 2),
            "recommendations": self._get_audio_recommendations(audio_score)
        }
        """

    def _analyze_motion(self, video_path: str, info: dict) -> dict:
        """Analyze motion quality - based on frame rate and smoothness"""
        metrics = self._extract_video_metrics(info)
        
        fps = metrics["fps"]
        
        # Frame rate score
        if fps >= 60:
            frame_score = 95
        elif fps >= 30:
            frame_score = 85
        elif fps >= 24:
            frame_score = 75
        else:
            frame_score = 60
        
        # Duration score
        duration = metrics["duration"]
        if 10 <= duration <= 600:  # 10 seconds to 10 minutes
            duration_score = 85
        elif 5 <= duration < 10 or 600 < duration <= 1800:  # 5-10 seconds or 10-30 minutes
            duration_score = 75
        else:
            duration_score = 65
        
        # Stability score
        stability_score = 80
        if HAS_OPENCV:
            try:
                cap = cv2.VideoCapture(video_path)
                prev_frame = None
                frame_count = 0
                movement_sum = 0
                
                while frame_count < 10:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize for faster processing
                    frame_small = cv2.resize(frame, (320, 240))
                    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate frame difference
                        diff = cv2.absdiff(prev_frame, gray)
                        movement = diff.mean()
                        movement_sum += movement
                    
                    prev_frame = gray
                    frame_count += 1
                
                cap.release()
                
                if frame_count > 1:
                    avg_movement = movement_sum / (frame_count - 1)
                    # Low motion = stable, high motion = jittery
                    if avg_movement < 10:
                        stability_score = 90
                    elif avg_movement < 30:
                        stability_score = 80
                    elif avg_movement < 60:
                        stability_score = 70
                    else:
                        stability_score = 60
            except Exception as e:
                print(f"[INFO] Stability analysis failed: {e}")
        
        motion_score = (frame_score * 0.5 + stability_score * 0.3 + duration_score * 0.2)
        
        return {
            "analysis_type": "motion",
            "metrics": {
                "fps": fps,
                "duration_seconds": round(duration, 1)
            },
            "scores": {
                "frame_rate": round(frame_score, 1),
                "stability": round(stability_score, 1),
                "duration": round(duration_score, 1)
            },
            "overall_motion_score": round(motion_score, 2),
            "recommendations": self._get_motion_recommendations(motion_score)
        }

    def _analyze_content(self, video_path: str, info: dict) -> dict:
        """Analyze content quality - using Qwen AI analysis"""
        try:
            file_size = info.get("file_size", 0)
            metrics = self._extract_video_metrics(info)
            
            # Estimate content complexity based on file size
            if file_size > 100 * 1024 * 1024:  # > 100MB
                complexity_score = 85
            elif file_size > 50 * 1024 * 1024:  # > 50MB
                complexity_score = 80
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                complexity_score = 75
            else:
                complexity_score = 65
            
            # Estimate content adequacy based on duration
            duration = metrics["duration"]
            if duration >= 30:  # Sufficient content length
                content_length_score = 85
            elif duration >= 10:
                content_length_score = 75
            else:
                content_length_score = 60
            
            # Try to call Qwen for content analysis
            analysis_text = self._call_qwen_for_analysis(video_path, info)

            # Default AI-derived scores
            coherence_score = 78
            engagement_score = 76

            # If we have AI analysis, try to parse structured JSON first,
            # otherwise fall back to simple keyword heuristics.
            if analysis_text:
                try:
                    parsed = json.loads(analysis_text)
                    if isinstance(parsed, dict):
                        if parsed.get("coherence") is not None:
                            coherence_score = int(parsed.get("coherence"))
                        if parsed.get("engagement") is not None:
                            engagement_score = int(parsed.get("engagement"))
                        # allow AI to influence complexity/length if provided
                        if parsed.get("complexity") is not None:
                            complexity_score = int(parsed.get("complexity"))
                        if parsed.get("length") is not None:
                            content_length_score = int(parsed.get("length"))
                except Exception:
                    # fallback: keyword heuristics
                    try:
                        txt = analysis_text.lower()
                        if "excellent" in txt:
                            coherence_score = 85
                            engagement_score = 85
                        elif "good" in txt:
                            coherence_score = 80
                            engagement_score = 80
                    except Exception:
                        pass

            # Compute AI component and heuristic component
            ai_component = (coherence_score + engagement_score) / 2.0
            heuristic_component = (complexity_score * 0.5 + content_length_score * 0.5)

            # New requirement: make Qwen AI score account for 80% of content_analysis
            content_score = ai_component * 0.8 + heuristic_component * 0.2

            return {
                "analysis_type": "content",
                "metrics": {
                    "file_size_mb": round(file_size / (1024 * 1024), 1),
                    "duration_seconds": round(duration, 1)
                },
                "scores": {
                    "coherence": round(coherence_score, 1),
                    "engagement": round(engagement_score, 1),
                    "complexity": round(complexity_score, 1),
                    "length": round(content_length_score, 1)
                },
                "overall_content_score": round(content_score, 2),
                "ai_analysis": analysis_text if analysis_text else "Unable to analyze content",
                "recommendations": self._get_content_recommendations(content_score)
            }
        except Exception as e:
            print(f"[INFO] Content analysis failed: {e}")
            return {
                "analysis_type": "content",
                "overall_content_score": 75.0,
                "error": str(e),
                "recommendations": ["Content analysis unavailable"]
            }

    def _call_qwen_for_analysis(self, video_path: str, info: dict) -> Optional[str]:
        """Call Qwen to analyze video content (based on metadata).

        Requires DashScope API key in environment variable `DASHSCOPE_API_KEY` or
        `dashscope.api_key` to be set. If not present, returns None.
        """
        try:
            # Ensure API key is configured
            if not dashscope.api_key:
                try:
                    from config_loader import get_dashscope_api_key
                    dashscope.api_key = get_dashscope_api_key()
                except (ImportError, FileNotFoundError, ValueError, KeyError):
                    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

            if not dashscope.api_key:
                print("[INFO] Qwen API key not found; skipping AI content analysis.")
                return None

            metrics = self._extract_video_metrics(info)
            file_name = Path(video_path).name

            prompt = f"""Analyze the content quality of this video. Video filename: {file_name}

Video technical parameters:
- Resolution: {metrics['width']}x{metrics['height']}
- Frame rate: {metrics['fps']} fps
- Duration: {metrics['duration']} seconds
- Bitrate: {metrics['bitrate']} kbps
- Encoding: {metrics['codec']}

Evaluate the video content for:
1. Coherence (logical clarity, smooth transitions)
2. Engagement (attractiveness, interestingness)
3. Professionalism (production quality)

Summarize your assessment in one sentence and include a short suggestion."""

            response = dashscope.Generation.call(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                result_format="message"
            )

            if response.status_code == HTTPStatus.OK and getattr(response, 'output', None):
                try:
                    return response.output.choices[0].message.content
                except Exception:
                    return None
        except Exception as e:
            print(f"[INFO] Qwen API call failed: {e}")

        return None

    def _analyze_overall(self, video_path: str, info: dict) -> dict:
        """Overall analysis - comprehensive assessment across all dimensions"""
        visual = self._analyze_visual_quality(video_path, info)
        # Audio analysis is disabled; keep placeholder in output
        audio = {"note": "Audio analysis disabled"}
        content = self._analyze_content(video_path, info)
        motion = self._analyze_motion(video_path, info)

        # Re-weight scores giving content a 60% weight as requested.
        # New weights: content 0.60, visual 0.25, motion 0.15
        overall_score = (
            visual["overall_visual_score"] * 0.25 +
            content["overall_content_score"] * 0.60 +
            motion["overall_motion_score"] * 0.15
        )
        
        # Return only numeric scores, no text recommendations
        return {
            "overall_score": round(overall_score, 2),
            "visual_score": round(visual.get("overall_visual_score", 0.0), 2),
            "content_score": round(content.get("overall_content_score", 0.0), 2),
            "motion_score": round(motion.get("overall_motion_score", 0.0), 2),
            "quality_level": self._get_quality_level(overall_score),
            "scores": {
                "visual": visual.get("scores", {}),
                "content": content.get("scores", {}),
                "motion": motion.get("scores", {})
            }
        }

    @staticmethod
    def _get_quality_level(score: float) -> str:
        """Get quality level"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    def _get_visual_recommendations(score: float) -> list:
        """Visual quality recommendations"""
        if score >= 85:
            return ["Video visual quality is excellent"]
        elif score >= 75:
            return ["Consider improving color grading", "Enhance contrast", "Maintain current quality"]
        else:
            return ["Upgrade to higher resolution", "Increase bitrate", "Better lighting setup needed"]

    @staticmethod
    def _get_audio_recommendations(score: float) -> list:
        """Audio quality recommendations"""
        if score >= 85:
            return ["Audio quality is excellent"]
        elif score >= 75:
            return ["Reduce background noise", "Consider higher sample rate"]
        else:
            return ["Add audio track if missing", "Use professional recording equipment", "Increase audio quality"]

    @staticmethod
    def _get_content_recommendations(score: float) -> list:
        """Content recommendations"""
        if score >= 85:
            return ["Content is engaging and coherent"]
        elif score >= 75:
            return ["Improve pacing", "Better narrative structure", "Enhance transitions"]
        else:
            return ["Restructure content flow", "Enhance storytelling elements", "Add more context"]

    @staticmethod
    def _get_motion_recommendations(score: float) -> list:
        """Motion recommendations"""
        if score >= 85:
            return ["Motion quality is smooth and stable"]
        elif score >= 75:
            return ["Use stabilization techniques", "Check frame rate settings"]
        else:
            return ["Improve frame rate", "Add video stabilization", "Reduce camera shake"]

    @staticmethod
    def _get_overall_recommendations(score: float) -> list:
        """Overall recommendations"""
        if score >= 85:
            return ["Video quality meets professional standards", "Ready for distribution"]
        elif score >= 75:
            return ["Minor improvements needed in multiple areas", "Good for most platforms"]
        else:
            return ["Significant improvements recommended", "Review all quality metrics", "Consider re-encoding"]


class VideoScoringAgent(Assistant):
    """Video scoring agent - video analysis assistant based on qwen-agent framework"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize video scoring agent
        
        Args:
            api_key: DashScope API key (optional, defaults to environment variable)
        """
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
        
        # Define system message
        system_message = """You are an expert video quality assessment specialist. 
Your task is to analyze videos and provide comprehensive quality scores and recommendations.

When analyzing a video:
1. Use the video_analyzer tool to assess different dimensions
2. Provide detailed scores for each aspect
3. Give actionable recommendations for improvement
4. Provide an overall quality rating

Always be objective and provide specific, constructive feedback."""

        # Initialize parent class
        tools = ["video_analyzer"]
        llm_config = {"model": "qwen-turbo"}
        super().__init__(
            llm=llm_config,
            name="Video Scoring Agent",
            description="Analyzes video quality and provides comprehensive scores",
            system_message=system_message,
            function_list=tools,
            **kwargs
        )

    def score_video(self, video_path: str) -> dict:
        """Score a video
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary containing scoring results
        """
        # Verify file exists
        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video file not found: {video_path}",
                "score": 0
            }
        
        try:
            # Run analysis
            response = self._run_analysis(video_path)
            return response
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "score": 0
            }

    def _run_analysis(self, video_path: str) -> dict:
        """Execute analysis task"""
        analyzer = VideoAnalyzer()
        result = analyzer.call(video_path, analysis_type="overall")
        result_dict = json.loads(result)
        
        return {
            "success": True,
            "video_path": video_path,
            "analysis": result_dict
        }


def score_video(video_path: str, api_key: Optional[str] = None) -> dict:
    """Quick function: score a video
    
    Args:
        video_path: Path to video file
        api_key: DashScope API key (optional)
    
    Returns:
        Dictionary with scores for each dimension and overall score
    
    Example:
        >>> result = score_video("path/to/video.mp4")
        >>> print(result["analysis"]["overall_score"])
    """
    # 支持直接传入本地路径或 HTTP(S) 视频 URL。
    # 如果是 URL，则先下载到临时文件再做分析。
    if not os.path.exists(video_path) and isinstance(video_path, str) and (
        video_path.startswith("http://") or video_path.startswith("https://")
    ):
        try:
            tmp_dir = tempfile.mkdtemp(prefix="video_score_")
            local_path = os.path.join(tmp_dir, "video.mp4")
            urllib.request.urlretrieve(video_path, local_path)
            video_path = local_path
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to download video from URL: {e}",
                "overall_score": 0,
            }

    analyzer = VideoAnalyzer()
    result = analyzer.call(video_path, analysis_type="overall")
    return json.loads(result)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = score_video(video_path)
        # Only output the overall score (numeric) when run from CLI
        try:
            overall = None
            if isinstance(result, dict):
                overall = result.get("overall_score") or result.get("analysis", {}).get("overall_score")
            if overall is None:
                # Fall back to printing 0 if not available
                print(0)
            else:
                print(overall)
        except Exception:
            print(0)
    else:
        print("Usage: python video_scorer.py <video_path>")
        print("\nExample:")
        print("  python video_scorer.py sample_video.mp4")
