# blockchain.py - add these functions (or replace existing helpers)

import json
import os
import time
import hashlib

CHAIN_PATH = os.path.join("data", "chain.json")  # adjust path if your chain file is elsewhere

def _load_chain_raw():
    """Return the raw loaded JSON (list or dict)."""
    if not os.path.exists(CHAIN_PATH):
        return []
    try:
        with open(CHAIN_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_chain_raw(raw):
    """Save raw chain object."""
    os.makedirs(os.path.dirname(CHAIN_PATH), exist_ok=True)
    with open(CHAIN_PATH, "w") as f:
        json.dump(raw, f, indent=2)

def normalize_chain():
    """
    Return the chain as a plain list of blocks.
    Accepts either:
      - {"chain": [ ... ]}
      - [ ... ]
    """
    raw = _load_chain_raw()
    if isinstance(raw, dict) and "chain" in raw and isinstance(raw["chain"], list):
        return raw["chain"]
    if isinstance(raw, list):
        return raw
    # unknown format — return empty
    return []

def save_normalized_chain(chain_list):
    """
    Save a normalized list into CHAIN_PATH; preserve original wrapper if possible.
    We will save as plain list for simplicity.
    """
    _save_chain_raw(chain_list)

def get_all_hashes():
    """Return list of stored document hashes (supports 'file_hash' or 'hash' keys)."""
    chain = normalize_chain()
    hashes = []
    for block in chain:
        if isinstance(block, dict):
            if "file_hash" in block:
                hashes.append(block["file_hash"])
            elif "hash" in block:
                hashes.append(block["hash"])
    return hashes

def create_block(file_hash: str, owner: str = None, extra: dict = None) -> dict:
    """
    Create a block dict (without persisting).
    extra may contain additional metadata to include in the block.
    """
    chain = normalize_chain()
    prev_hash = chain[-1].get("block_hash") if chain else "0"
    block = {
        "index": len(chain),
        "timestamp": int(time.time()),
        "file_hash": file_hash,
        "prev_hash": prev_hash,
    }
    if owner:
        block["owner"] = owner
    if extra:
        block.update(extra)
    # deterministic block hash
    block_hash = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
    block["block_hash"] = block_hash
    return block

def add_block(file_hash: str, owner: str = None, extra: dict = None) -> dict:
    """
    Append a new block to the chain and save it. Returns the new block.
    """
    chain = normalize_chain()
    block = create_block(file_hash, owner=owner, extra=extra)
    chain.append(block)
    save_normalized_chain(chain)
    return block

def find_blocks_by_filehash(file_hash: str):
    """Return list of blocks matching a document file_hash."""
    chain = normalize_chain()
    return [b for b in chain if isinstance(b, dict) and (b.get("file_hash") == file_hash or b.get("hash") == file_hash)]
