"""
Logging AgenTool - Provides structured logging with multiple output formats.

This toolkit provides a comprehensive logging interface with support for
different log levels, structured data, multiple output formats (console, file),
and log rotation capabilities.

Features:
- Multiple log levels (DEBUG, INFO, WARN, ERROR, CRITICAL)
- Structured logging with JSON support
- Multiple output destinations (console, file, both)
- Log file rotation and cleanup
- Logger namespacing
- Custom log formatting
- Integration with storage_fs for file operations

Example Usage:
    >>> from agentoolkit.system.logging import create_logging_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_logging_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('logging', {
    ...     "operation": "log",
    ...     "level": "INFO",
    ...     "message": "User logged in",
    ...     "data": {"user_id": "123", "timestamp": "2024-01-01T12:00:00Z"}
    ... })
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class LoggingInput(BaseOperationInput):
    """Input schema for logging operations."""
    operation: Literal['log', 'configure', 'get_logs', 'clear_logs', 'rotate_logs'] = Field(
        description="The logging operation to perform"
    )
    level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'] = Field(
        default="INFO", description="Log level"
    )
    message: Optional[str] = Field(None, description="Log message")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured log data")
    logger_name: str = Field(default="default", description="Logger name/namespace")
    output: Literal['console', 'file', 'both'] = Field(default="console", description="Log output destination")
    file_path: Optional[str] = Field(None, description="Log file path (for file output)")
    format: Literal['text', 'json'] = Field(default="text", description="Log format")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes (10MB default)")
    max_files: int = Field(default=5, description="Maximum number of rotated log files to keep")


class LoggingOutput(BaseModel):
    """Structured output for logging operations."""
    operation: str = Field(description="The operation that was performed")
    logger_name: str = Field(description="The logger name used")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


# Global logging configuration storage
_logging_config: Dict[str, Dict[str, Any]] = {}


def _get_log_level_priority(level: str) -> int:
    """Get numeric priority for log level."""
    priorities = {
        'DEBUG': 10,
        'INFO': 20,
        'WARN': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    return priorities.get(level, 20)


def _format_log_entry(level: str, message: str, data: Optional[Dict[str, Any]], 
                     logger_name: str, format_type: str) -> str:
    """
    Format a log entry according to the specified format.
    
    Args:
        level: Log level
        message: Log message
        data: Structured data
        logger_name: Logger name
        format_type: Format type (text or json)
        
    Returns:
        Formatted log entry string
    """
    timestamp = datetime.now().isoformat()
    
    if format_type == 'json':
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "logger": logger_name,
            "message": message
        }
        if data:
            log_entry["data"] = data
        return json.dumps(log_entry)
    else:
        # Text format
        base_msg = f"[{timestamp}] {level:<8} [{logger_name}] {message}"
        if data:
            data_str = json.dumps(data, separators=(',', ':'))
            return f"{base_msg} | {data_str}"
        return base_msg


async def logging_log(ctx: RunContext[Any], level: str, message: str, data: Optional[Dict[str, Any]], 
                     logger_name: str, output: str, file_path: Optional[str], format: str) -> LoggingOutput:
    """
    Log a message with structured data.
    
    Args:
        ctx: Runtime context
        level: Log level
        message: Log message
        data: Optional structured data
        logger_name: Logger name
        output: Output destination
        file_path: Optional file path for file output
        format: Log format
        
    Returns:
        LoggingOutput with the log operation result
    """
    try:
        # Get logger configuration
        logger_config = _logging_config.get(logger_name, {
            "min_level": "DEBUG",
            "output": "console",
            "file_path": "/tmp/app.log",
            "format": "text"
        })
        
        # Use provided values or fall back to logger config
        actual_output = output if output != "console" else logger_config.get("output", "console")
        actual_file_path = file_path or logger_config.get("file_path", "/tmp/app.log")
        actual_format = format if format != "text" else logger_config.get("format", "text")
        min_level = logger_config.get("min_level", "DEBUG")
        
        # Check if log level meets minimum threshold
        if _get_log_level_priority(level) < _get_log_level_priority(min_level):
            return LoggingOutput(
                operation="log",
                logger_name=logger_name,
                message=f"Log level {level} below minimum threshold {min_level}, skipped",
                data={"skipped": True, "level": level, "min_level": min_level}
            )
        
        # Format the log entry
        log_entry = _format_log_entry(level, message, data, logger_name, actual_format)
        
        outputs_written = []
        
        # Console output
        if actual_output in ['console', 'both']:
            print(log_entry)
            outputs_written.append("console")
        
        # File output
        if actual_output in ['file', 'both']:
            try:
                injector = get_injector()
                
                # Check if file rotation is needed
                file_exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": actual_file_path
                })
                
                if hasattr(file_exists_result, 'output'):
                    exists_data = json.loads(file_exists_result.output)
                else:
                    exists_data = file_exists_result
                
                # If file exists and might need rotation
                if exists_data.get("data", {}).get("exists", False):
                    file_size = exists_data["data"].get("size", 0)
                    max_size = logger_config.get("max_file_size", 10485760)  # 10MB default
                    
                    if file_size >= max_size:
                        # Rotate the log file
                        await _rotate_log_file(actual_file_path, logger_config.get("max_files", 5))
                
                # Append to log file
                append_result = await injector.run('storage_fs', {
                    "operation": "append",
                    "path": actual_file_path,
                    "content": log_entry + "\n",
                    "create_parents": True
                })
                
                outputs_written.append("file")
                
            except Exception as file_error:
                raise IOError(f"Error writing to log file: {str(file_error)}") from file_error
        
        return LoggingOutput(
            operation="log",
            logger_name=logger_name,
            message=f"Successfully logged {level} message to {', '.join(outputs_written)}",
            data={
                "level": level,
                "outputs": outputs_written,
                "format": actual_format,
                "entry_length": len(log_entry)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error logging message: {str(e)}") from e


async def _rotate_log_file(file_path: str, max_files: int) -> None:
    """
    Rotate log files when they get too large.
    
    Args:
        file_path: Path to the log file
        max_files: Maximum number of rotated files to keep
    """
    injector = get_injector()
    
    try:
        # Move current log to .1
        base_path = file_path
        rotated_path = f"{base_path}.1"
        
        # Read current content
        try:
            read_result = await injector.run('storage_fs', {
                "operation": "read",
                "path": base_path
            })
            
            if hasattr(read_result, 'output'):
                read_data = json.loads(read_result.output)
            else:
                read_data = read_result
            
            content = read_data["data"]["content"]
            
            # Write to rotated file
            await injector.run('storage_fs', {
                "operation": "write",
                "path": rotated_path,
                "content": content,
                "create_parents": True
            })
            
            # Clear original file
            await injector.run('storage_fs', {
                "operation": "write",
                "path": base_path,
                "content": "",
                "create_parents": True
            })
        except FileNotFoundError:
            # File doesn't exist, nothing to rotate
            pass
            
            # Remove old rotated files beyond max_files
            for i in range(max_files + 1, max_files + 10):  # Check some extra files
                old_file = f"{base_path}.{i}"
                await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": old_file
                })
    
    except Exception:
        # If rotation fails, continue logging to prevent loss
        pass


async def logging_configure(ctx: RunContext[Any], logger_name: str, level: str, output: str, 
                          file_path: Optional[str], format: str, max_file_size: int, max_files: int) -> LoggingOutput:
    """
    Configure a logger.
    
    Args:
        ctx: Runtime context
        logger_name: Logger name to configure
        level: Minimum log level
        output: Output destination
        file_path: Log file path
        format: Log format
        max_file_size: Maximum file size before rotation
        max_files: Maximum number of rotated files
        
    Returns:
        LoggingOutput with the configuration result
    """
    try:
        config = {
            "min_level": level,
            "output": output,
            "format": format,
            "max_file_size": max_file_size,
            "max_files": max_files
        }
        
        if file_path:
            config["file_path"] = file_path
        
        _logging_config[logger_name] = config
        
        return LoggingOutput(
            operation="configure",
            logger_name=logger_name,
            message=f"Successfully configured logger '{logger_name}'",
            data={
                "configuration": config,
                "total_loggers": len(_logging_config)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error configuring logger '{logger_name}': {str(e)}") from e


async def logging_get_logs(ctx: RunContext[Any], logger_name: str, file_path: Optional[str], 
                          level: Optional[str]) -> LoggingOutput:
    """
    Get recent log entries.
    
    Args:
        ctx: Runtime context
        logger_name: Logger name
        file_path: Optional file path to read from
        level: Optional minimum level filter
        
    Returns:
        LoggingOutput with the log entries
    """
    try:
        logger_config = _logging_config.get(logger_name, {})
        actual_file_path = file_path or logger_config.get("file_path", "/tmp/app.log")
        
        injector = get_injector()
        
        # Read log file
        try:
            read_result = await injector.run('storage_fs', {
                "operation": "read",
                "path": actual_file_path
            })
            
            if hasattr(read_result, 'output'):
                read_data = json.loads(read_result.output)
            else:
                read_data = read_result
            
            content = read_data["data"]["content"]
        except FileNotFoundError:
            # Log file doesn't exist yet, return empty results
            return LoggingOutput(
                operation="get_logs",
                logger_name=logger_name,
                message=f"No log file found at {actual_file_path}",
                data={"entries": [], "count": 0}
            )
        lines = content.strip().split('\n') if content.strip() else []
        
        # Parse log entries
        entries = []
        for line in lines:
            if not line.strip():
                continue
                
            try:
                # Try to parse as JSON first
                entry = json.loads(line)
                if level and _get_log_level_priority(entry.get("level", "INFO")) < _get_log_level_priority(level):
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                # Parse text format
                if level:
                    # Simple level check for text format
                    if f" {level} " not in line and not any(f" {lvl} " in line for lvl in ['ERROR', 'CRITICAL'] 
                                                           if _get_log_level_priority(lvl) >= _get_log_level_priority(level)):
                        continue
                entries.append({"raw": line})
        
        # Get recent entries (last 100)
        recent_entries = entries[-100:] if len(entries) > 100 else entries
        
        return LoggingOutput(
            operation="get_logs",
            logger_name=logger_name,
            message=f"Retrieved {len(recent_entries)} log entries",
            data={
                "entries": recent_entries,
                "count": len(recent_entries),
                "total_lines": len(lines),
                "file_path": actual_file_path
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error retrieving logs: {str(e)}") from e


async def logging_clear_logs(ctx: RunContext[Any], logger_name: str, file_path: Optional[str]) -> LoggingOutput:
    """
    Clear log files.
    
    Args:
        ctx: Runtime context
        logger_name: Logger name
        file_path: Optional file path to clear
        
    Returns:
        LoggingOutput with the clear operation result
    """
    try:
        logger_config = _logging_config.get(logger_name, {})
        actual_file_path = file_path or logger_config.get("file_path", "/tmp/app.log")
        
        injector = get_injector()
        
        # Clear the main log file
        clear_result = await injector.run('storage_fs', {
            "operation": "write",
            "path": actual_file_path,
            "content": "",
            "create_parents": True
        })
        
        # Clear rotated log files
        cleared_files = [actual_file_path]
        for i in range(1, 10):  # Clear up to 10 rotated files
            rotated_path = f"{actual_file_path}.{i}"
            delete_result = await injector.run('storage_fs', {
                "operation": "delete",
                "path": rotated_path
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            if delete_data.get("data", {}).get("deleted", False):
                cleared_files.append(rotated_path)
        
        return LoggingOutput(
            operation="clear_logs",
            logger_name=logger_name,
            message=f"Cleared {len(cleared_files)} log files",
            data={
                "cleared_files": cleared_files,
                "count": len(cleared_files)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error clearing logs: {str(e)}") from e


def create_logging_agent():
    """
    Create and return the logging AgenTool.
    
    Returns:
        Agent configured for logging operations
    """
    logging_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'log': ('logging_log', lambda x: {
                'level': x.level, 'message': x.message, 'data': x.data,
                'logger_name': x.logger_name, 'output': x.output,
                'file_path': x.file_path, 'format': x.format
            }),
            'configure': ('logging_configure', lambda x: {
                'logger_name': x.logger_name, 'level': x.level, 'output': x.output,
                'file_path': x.file_path, 'format': x.format,
                'max_file_size': x.max_file_size, 'max_files': x.max_files
            }),
            'get_logs': ('logging_get_logs', lambda x: {
                'logger_name': x.logger_name, 'file_path': x.file_path, 'level': x.level
            }),
            'clear_logs': ('logging_clear_logs', lambda x: {
                'logger_name': x.logger_name, 'file_path': x.file_path
            }),
        }
    )
    
    return create_agentool(
        name='logging',
        input_schema=LoggingInput,
        routing_config=logging_routing,
        tools=[logging_log, logging_configure, logging_get_logs, logging_clear_logs],
        output_type=LoggingOutput,
        system_prompt="Handle structured logging with multiple output formats and log rotation.",
        description="Structured logging with multiple outputs, log rotation, and level filtering",
        version="1.0.0",
        tags=["logging", "monitoring", "observability", "structured-data"],
        dependencies=["storage_fs", "metrics"],
        examples=[
            {
                "description": "Log an info message with structured data",
                "input": {
                    "operation": "log",
                    "level": "INFO",
                    "message": "User logged in",
                    "data": {"user_id": "123", "ip": "192.168.1.1"},
                    "logger_name": "auth"
                },
                "output": {
                    "operation": "log",
                    "logger_name": "auth",
                    "message": "Successfully logged INFO message to console"
                }
            },
            {
                "description": "Configure a logger for file output",
                "input": {
                    "operation": "configure",
                    "logger_name": "api",
                    "level": "WARN",
                    "output": "file",
                    "file_path": "/var/log/api.log",
                    "format": "json"
                },
                "output": {
                    "operation": "configure",
                    "logger_name": "api",
                    "message": "Successfully configured logger 'api'"
                }
            },
            {
                "description": "Get recent log entries",
                "input": {
                    "operation": "get_logs",
                    "logger_name": "api",
                    "level": "ERROR"
                },
                "output": {
                    "operation": "get_logs",
                    "logger_name": "api",
                    "message": "Retrieved 5 log entries",
                    "data": {"entries": [], "count": 5}
                }
            }
        ]
    )


# Create the agent instance
agent = create_logging_agent()