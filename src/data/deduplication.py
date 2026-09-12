import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm


def sha256_of_file(path: str | Path, chunk_size: int = 65536) -> str:
    """
    Compute cryptographic SHA-256 hash of an image file using 64 KB chunks.
    Maintains low memory footprint regardless of dataset scale.
    """
    path_str = str(path)
    if path_str == "NA" or not os.path.exists(path_str):
        return "NA"
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_dataframe(
    df: pd.DataFrame,
    hash_cache_path: str | Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Perform exact cryptographic (SHA-256) deduplication across images.
    Returns deduplicated DataFrame and dictionary of duplicate removal counts.
    """
    cached_map: Dict[str, str] = {}
    if hash_cache_path and Path(hash_cache_path).exists():
        try:
            cdf = pd.read_csv(hash_cache_path)
            cached_map = dict(zip(cdf['image_path'], cdf['sha256']))
        except Exception:
            cached_map = {}

    needed = [p for p in df['image_path'].unique() if p != 'NA' and p not in cached_map and os.path.exists(p)]
    if needed:
        from concurrent.futures import ThreadPoolExecutor
        def worker(p):
            return p, sha256_of_file(p)

        with ThreadPoolExecutor(max_workers=8) as ex:
            for p, h in tqdm(ex.map(worker, needed), total=len(needed), desc="Computing SHA-256"):
                cached_map[p] = h

        if hash_cache_path:
            try:
                pd.DataFrame(list(cached_map.items()), columns=['image_path', 'sha256']).to_csv(hash_cache_path, index=False)
            except Exception:
                pass

    df_copy = df.copy()
    df_copy['sha256'] = df_copy['image_path'].map(cached_map).fillna('NA')

    before_count = len(df_copy)
    valid_hashes = df_copy[df_copy['sha256'] != 'NA'].drop_duplicates(subset='sha256', keep='first')
    na_hashes = df_copy[df_copy['sha256'] == 'NA']
    deduped_df = pd.concat([valid_hashes, na_hashes], ignore_index=True)
    exact_removed = before_count - len(deduped_df)

    stats = {
        "sha256_duplicate_images_removed": exact_removed,
        "confirmed_phash_near_duplicate_images_removed": 0,
        "retained_records": len(deduped_df)
    }
    return deduped_df, stats
