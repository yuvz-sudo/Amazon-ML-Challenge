"""Text preprocessing for business names and addresses."""
import re
import pandas as pd


def normalize_name(name):
    """Normalize business name for matching."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()

    # Remove common noise characters
    s = re.sub(r'[<>\-]+', ' ', s)

    # Standardize legal suffixes
    replacements = {
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bllc\b': 'limited liability company',
        r'\bllp\b': 'limited liability partnership',
        r'\bco\b': 'company',
        r'\bsarl\b': 'sarl',
        r'\bsas\b': 'sas',
        r'\bs\.?a\.?s\b': 'sas',
        r'\bs\.?a\.?r\.?l\b': 'sarl',
        r'\bsci\b': 'sci',
        r'\b&\b': 'and',
        r'\bdba\b': '',
    }
    for pattern, replacement in replacements.items():
        s = re.sub(pattern, replacement, s)

    # Remove punctuation except alphanumeric and spaces
    s = re.sub(r'[^\w\s]', ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_address(addr):
    """Normalize business address for matching."""
    if pd.isna(addr):
        return ""
    s = str(addr).lower().strip()

    # Standardize common abbreviations
    replacements = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bcir\b': 'circle',
        r'\bhwy\b': 'highway',
        r'\bpkwy\b': 'parkway',
        r'\bapt\b': 'apartment',
        r'\bste\b': 'suite',
        r'\bfl\b': 'floor',
        r'\bno\b': 'number',
        r'\bkh\b': 'khasra',
        r'\bdist\b': 'district',
        r'\btq\b': 'taluka',
        r'\br\.\b': 'rue',
    }
    for pattern, replacement in replacements.items():
        s = re.sub(pattern, replacement, s)

    # Remove punctuation
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def preprocess_df(df):
    """Apply all preprocessing to a dataframe."""
    df = df.copy()
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    # Combined text for TF-IDF blocking
    df['combined_text'] = df['name_norm'] + ' ' + df['addr_norm']
    return df
