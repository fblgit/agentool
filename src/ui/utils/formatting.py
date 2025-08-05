"""
Formatting utilities for UI display.

This module provides helper functions for formatting various
data types for display in the Streamlit UI.
"""

from datetime import datetime, timedelta
from typing import Any, Optional, Union
import json


def format_timestamp(timestamp: Union[datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp for display.
    
    Args:
        timestamp: Datetime object or ISO string
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        # Parse ISO format
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return timestamp.strftime(format_str)


def format_duration(seconds: float, precision: int = 1) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Decimal places for seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.{precision}f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.{precision}f}s"


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_number(num: Union[int, float], decimals: int = 0) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if decimals == 0:
        return f"{int(num):,}"
    else:
        return f"{num:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as percentage.
    
    Args:
        value: Decimal value (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_json(data: Any, indent: int = 2, max_length: Optional[int] = None) -> str:
    """
    Format JSON data for display.
    
    Args:
        data: Data to format
        indent: Indentation level
        max_length: Maximum string length (truncate if longer)
        
    Returns:
        Formatted JSON string
    """
    json_str = json.dumps(data, indent=indent, default=str)
    
    if max_length and len(json_str) > max_length:
        return json_str[:max_length] + "..."
    
    return json_str


def format_error(error: Union[Exception, str], include_type: bool = True) -> str:
    """
    Format an error for display.
    
    Args:
        error: Exception or error string
        include_type: Whether to include error type
        
    Returns:
        Formatted error string
    """
    if isinstance(error, Exception):
        if include_type:
            return f"{type(error).__name__}: {str(error)}"
        else:
            return str(error)
    else:
        return str(error)


def format_list(items: list, max_items: int = 10, separator: str = ", ") -> str:
    """
    Format a list for display with optional truncation.
    
    Args:
        items: List of items
        max_items: Maximum items to show
        separator: Item separator
        
    Returns:
        Formatted list string
    """
    if len(items) <= max_items:
        return separator.join(str(item) for item in items)
    else:
        displayed = separator.join(str(item) for item in items[:max_items])
        remaining = len(items) - max_items
        return f"{displayed} ... and {remaining} more"


def format_code_snippet(code: str, max_lines: int = 10, line_numbers: bool = True) -> str:
    """
    Format a code snippet with optional truncation.
    
    Args:
        code: Code string
        max_lines: Maximum lines to show
        line_numbers: Whether to add line numbers
        
    Returns:
        Formatted code snippet
    """
    lines = code.strip().split('\n')
    
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    else:
        truncated = False
    
    if line_numbers:
        width = len(str(len(lines)))
        formatted_lines = []
        for i, line in enumerate(lines, 1):
            formatted_lines.append(f"{str(i).rjust(width)} | {line}")
        result = '\n'.join(formatted_lines)
    else:
        result = '\n'.join(lines)
    
    if truncated:
        result += f"\n... ({len(lines) - max_lines} more lines)"
    
    return result


def format_status(status: str) -> str:
    """
    Format a status string with emoji.
    
    Args:
        status: Status string
        
    Returns:
        Formatted status with emoji
    """
    status_emojis = {
        'pending': '⏳',
        'running': '🔄',
        'completed': '✅',
        'success': '✅',
        'failed': '❌',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'debug': '🐛'
    }
    
    emoji = status_emojis.get(status.lower(), '📌')
    return f"{emoji} {status.title()}"


def truncate_middle(text: str, max_length: int = 50) -> str:
    """
    Truncate text in the middle, preserving start and end.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Calculate how much to keep from each end
    keep_length = (max_length - 3) // 2  # -3 for "..."
    
    return f"{text[:keep_length]}...{text[-keep_length:]}"


def format_key_value_pairs(pairs: dict, separator: str = ": ", max_key_length: int = 20) -> str:
    """
    Format dictionary as key-value pairs.
    
    Args:
        pairs: Dictionary of key-value pairs
        separator: Separator between key and value
        max_key_length: Maximum key length for alignment
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in pairs.items():
        key_str = str(key).ljust(max_key_length)
        if len(key_str) > max_key_length:
            key_str = truncate_middle(key_str, max_key_length)
        lines.append(f"{key_str}{separator}{value}")
    
    return '\n'.join(lines)