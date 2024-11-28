def convert_change_percent(value):
    if isinstance(value, str):
        value = value.replace('%', '')  # Remove percentage sign
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000  # Convert millions to numeric
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000  # Convert thousands to numeric
        else:
            return float(value)  # Default conversion
    return value