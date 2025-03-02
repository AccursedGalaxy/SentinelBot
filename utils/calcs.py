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


def format_volume(volume_native, price=None):
    """
    Format volume to be more readable, including USD value if price is provided.

    Args:
        volume_native: The volume in native token units
        price: The price of the token in USD

    Returns:
        Formatted volume string with USD value if price is available
    """
    if volume_native is None:
        return "Unknown"

    # Format the native volume
    if volume_native >= 1_000_000_000:  # Billions
        formatted_native = f"{volume_native / 1_000_000_000:.2f}B"
    elif volume_native >= 1_000_000:  # Millions
        formatted_native = f"{volume_native / 1_000_000:.2f}M"
    elif volume_native >= 1_000:  # Thousands
        formatted_native = f"{volume_native / 1_000:.2f}K"
    else:
        formatted_native = f"{volume_native:.2f}"

    # Calculate and format USD value if price is provided
    if price is not None:
        usd_volume = volume_native * price
        if usd_volume >= 1_000_000_000:  # Billions
            formatted_usd = f"${usd_volume / 1_000_000_000:.2f}B"
        elif usd_volume >= 1_000_000:  # Millions
            formatted_usd = f"${usd_volume / 1_000_000:.2f}M"
        elif usd_volume >= 1_000:  # Thousands
            formatted_usd = f"${usd_volume / 1_000:.2f}K"
        else:
            formatted_usd = f"${usd_volume:.2f}"

        return f"{formatted_native} (≈{formatted_usd})"

    return formatted_native
