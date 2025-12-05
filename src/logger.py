import logging
import sys
import os
from typing import Optional


class ColorCodes:
    """ANSI color codes for terminal output"""
    # Reset
    RESET = '\033[0m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bold colors
    BOLD_RED = '\033[1;31m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_YELLOW = '\033[1;33m'
    BOLD_BLUE = '\033[1;34m'
    BOLD_MAGENTA = '\033[1;35m'
    BOLD_CYAN = '\033[1;36m'
    BOLD_WHITE = '\033[1;37m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding based on log level"""
    
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.CYAN,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.BOLD_RED + ColorCodes.BG_RED,
    }
    
    LEVEL_ICONS = {
        logging.DEBUG: "🔍",
        logging.INFO: "✓",
        logging.WARNING: "⚠️",
        logging.ERROR: "✗",
        logging.CRITICAL: "🔥",
    }
    
    def __init__(self, fmt: Optional[str] = None, use_icons: bool = True, use_colors: bool = True):
        super().__init__(fmt)
        self.use_icons = use_icons
        self.use_colors = use_colors
        
    def format(self, record):
        # Add icon if enabled
        if self.use_icons:
            icon = self.LEVEL_ICONS.get(record.levelno, "•")
            record.icon = icon
        else:
            record.icon = ""
            
        # Add color if enabled and output supports it
        if self.use_colors and sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, "")
            reset = ColorCodes.RESET
            
            # Color the level name
            record.levelname = f"{color}{record.levelname}{reset}"
            
            # Format the message
            formatted = super().format(record)
            
            # Dim the timestamp/metadata
            if hasattr(record, 'asctime'):
                formatted = formatted.replace(
                    record.asctime, 
                    f"{ColorCodes.DIM}{record.asctime}{reset}"
                )
                
            return formatted
        else:
            return super().format(record)


class ComponentFormatter(ColoredFormatter):
    """Enhanced formatter with component-specific colors"""
    
    COMPONENT_COLORS = {
        "SemanticRouter": ColorCodes.BOLD_CYAN,
        # "Router": ColorCodes.BOLD_CYAN,
        "DataGen": ColorCodes.BOLD_BLUE,
        "Validator": ColorCodes.BOLD_GREEN,
        "Exporter": ColorCodes.BOLD_YELLOW,
    }

    def format(self, record):
        # Add component color
        if self.use_colors and sys.stdout.isatty():
            component_color = self.COMPONENT_COLORS.get(record.name, ColorCodes.WHITE)
            record.name_colored = f"{component_color}{record.name}{ColorCodes.RESET}"
        else:
            record.name_colored = record.name
            
        return super().format(record)


def setup_logger(
    name: str = "SyntheticGen", 
    log_file: str = "generation.log",
    use_colors: bool = True,
    use_icons: bool = True,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Sets up a logger with color-coded console output and detailed file logging.
    
    Args:
        name: Logger name (also used for component color coding)
        log_file: Filename for detailed logs
        use_colors: Enable ANSI color codes in console output
        use_icons: Enable emoji icons for log levels
        console_level: Minimum level for console output (default: INFO)
        file_level: Minimum level for file output (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate logs if function is called multiple times
    if logger.handlers:
        return logger

    # 1. Console Handler (Color-coded with icons)
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(console_level)
    
    # Use colored formatter for console
    c_format = ComponentFormatter(
        fmt='%(icon)s %(name_colored)s %(levelname)s: %(message)s',
        use_icons=use_icons,
        use_colors=use_colors
    )
    c_handler.setFormatter(c_format)

    # 2. File Handler (Detailed debugging, no colors)
    os.makedirs("logs", exist_ok=True)
    f_handler = logging.FileHandler(os.path.join("logs", log_file))
    f_handler.setLevel(file_level)
    
    # Plain formatter for file (no colors/icons in logs)
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 60):
    """
    Log a colored section header.
    
    Example:
        ============================================================
                          STARTING GENERATION
        ============================================================
    """
    border = char * width
    logger.info(f"\n{ColorCodes.BOLD_CYAN}{border}{ColorCodes.RESET}")
    logger.info(f"{ColorCodes.BOLD_WHITE}{title.center(width)}{ColorCodes.RESET}")
    logger.info(f"{ColorCodes.BOLD_CYAN}{border}{ColorCodes.RESET}\n")


def log_progress(logger: logging.Logger, current: int, total: int, prefix: str = "Progress"):
    """
    Log a colored progress indicator.
    
    Example:
        ✓ Progress: [████████░░] 80% (800/1000)
    """
    percentage = (current / total) * 100
    filled = int(percentage / 10)
    bar = "█" * filled + "░" * (10 - filled)
    
    color = ColorCodes.GREEN if percentage == 100 else ColorCodes.CYAN
    logger.info(
        f"{color}{prefix}: [{bar}] {percentage:.1f}% ({current}/{total}){ColorCodes.RESET}"
    )


def log_tool_call(logger: logging.Logger, tool: str, args: dict):
    """
    Log a tool invocation with color coding.
    
    Example:
        🔧 TOOL: codebase_search | args: {"query": "auth_handler", "scope": "backend/"}
    """
    tool_colors = {
        "codebase_search": ColorCodes.BLUE,
        "sandbox_exec": ColorCodes.GREEN,
        "file_manager": ColorCodes.YELLOW,
        "ask_human": ColorCodes.MAGENTA,
    }
    
    color = tool_colors.get(tool, ColorCodes.WHITE)
    args_str = str(args)[:100]  # Truncate long args
    
    logger.info(
        f"{color}🔧 TOOL: {tool}{ColorCodes.RESET} | "
        f"{ColorCodes.DIM}args: {args_str}{ColorCodes.RESET}"
    )


def log_metrics(logger: logging.Logger, metrics: dict):
    """
    Log metrics in a formatted, color-coded table.
    
    Example:
        📊 METRICS:
           • samples_generated: 1000
           • validation_errors: 3
           • duration: 45.2s
    """
    logger.info(f"\n{ColorCodes.BOLD_BLUE}📊 METRICS:{ColorCodes.RESET}")
    for key, value in metrics.items():
        logger.info(f"   {ColorCodes.DIM}•{ColorCodes.RESET} {key}: {ColorCodes.BOLD_WHITE}{value}{ColorCodes.RESET}")


# Singleton instance with default settings
logger = setup_logger()


# Example usage
if __name__ == "__main__":
    # Demo different log levels
    logger.debug("This is a debug message")
    logger.info("Generation started successfully")
    logger.warning("Low diversity detected in persona distribution")
    logger.error("Failed to parse JSON schema")
    logger.critical("Out of memory - aborting generation")
    
    # Demo section headers
    log_section(logger, "STARTING GENERATION")
    
    # Demo progress
    for i in range(0, 101, 20):
        log_progress(logger, i, 100)
    
    # Demo tool call
    log_tool_call(logger, "codebase_search", {"query": "auth_handler", "scope": "backend/"})
    
    # Demo metrics
    log_metrics(logger, {
        "samples_generated": 1000,
        "validation_errors": 3,
        "duration_seconds": 45.2
    })
