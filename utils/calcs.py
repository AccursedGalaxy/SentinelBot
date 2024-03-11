def format_number(value):
    """Format a number to a string with appropriate precision based on its size."""
    if value == 0:
        return "0"
    elif abs(value) < 1e-9:
        return f"{value:.10f}"
    elif abs(value) < 1e-6:
        return f"{value:.8f}"
    elif abs(value) < 1e-3:
        return f"{value:.6f}"
    else:
        return f"{value:.2f}"


def format_currency(value: float) -> str:
    """Format a float as a currency string, converting to a more readable format."""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.1f}T"
    elif value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"
