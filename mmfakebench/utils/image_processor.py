"""Image processing utilities for the MMFakeBench toolkit.

This module provides functions for encoding, processing, and validating
images for use with multimodal language models.
"""

import base64
import logging
import os
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError


def encode_image(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Encode an image to base64, handling potential errors and converting formats.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_encoded_string, mime_type) or (None, None) if failed
    """
    try:
        # Check if the path exists and is a file
        if not os.path.isfile(image_path):
            logging.error(f"Image file not found or is not a file: {image_path}")
            return None, None

        with Image.open(image_path) as img:
            # Basic check for valid image format before extensive processing
            img.verify()  # Verifies headers, doesn't load full image data

        # Re-open after verify
        with Image.open(image_path) as img:
            logging.debug(f"Processing image: {image_path} (Mode: {img.mode}, Format: {img.format})")
            
            # Convert formats like P, LA, RGBA with transparency to RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                # If P mode with transparency, convert via RGBA
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')  # Convert other modes like L, CMYK etc. to RGB

            output_buffer = BytesIO()
            img.save(output_buffer, format='JPEG', quality=90)  # Save as JPEG for consistency
            image_data = output_buffer.getvalue()

            mime_type = "image/jpeg"
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image, mime_type

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None, None
    except UnidentifiedImageError:
        logging.error(f"Cannot identify image file (possibly corrupt or unsupported format): {image_path}")
        return None, None
    except Exception as e:
        logging.error(f"Image processing failed for {image_path}: {str(e)}")
        return None, None


def validate_image(image_path: str) -> bool:
    """Validate if an image file is readable and supported.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        if not os.path.isfile(image_path):
            return False
            
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> Optional[dict]:
    """Get information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information or None if failed
    """
    try:
        if not os.path.isfile(image_path):
            return None
            
        with Image.open(image_path) as img:
            return {
                'path': image_path,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': os.path.getsize(image_path)
            }
    except Exception as e:
        logging.error(f"Failed to get image info for {image_path}: {e}")
        return None


def resize_image(image_path: str, 
                max_width: int = 1024, 
                max_height: int = 1024, 
                quality: int = 90) -> Tuple[Optional[str], Optional[str]]:
    """Resize and encode an image with size constraints.
    
    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        quality: JPEG quality (1-100)
        
    Returns:
        Tuple of (base64_encoded_string, mime_type) or (None, None) if failed
    """
    try:
        if not os.path.isfile(image_path):
            logging.error(f"Image file not found: {image_path}")
            return None, None

        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if necessary
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                logging.debug(f"Resized image from original size to {img.size}")

            output_buffer = BytesIO()
            img.save(output_buffer, format='JPEG', quality=quality)
            image_data = output_buffer.getvalue()

            mime_type = "image/jpeg"
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image, mime_type

    except Exception as e:
        logging.error(f"Image resize/encoding failed for {image_path}: {str(e)}")
        return None, None


def batch_validate_images(image_paths: list) -> dict:
    """Validate multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': [],
        'invalid': [],
        'total': len(image_paths),
        'valid_count': 0,
        'invalid_count': 0
    }
    
    for image_path in image_paths:
        if validate_image(image_path):
            results['valid'].append(image_path)
            results['valid_count'] += 1
        else:
            results['invalid'].append(image_path)
            results['invalid_count'] += 1
    
    return results