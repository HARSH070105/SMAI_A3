"""
Downloads additional images from Wikimedia Commons for under-represented classes.
Adds them to data/24_monuments/Indian-monuments/images/train/{class}/

Prioritises classes below --target total images (default 200).
Run AFTER prepare_dataset.py.

Usage:
  python scripts/expand_dataset.py
  python scripts/expand_dataset.py --target 250 --per-request 50
"""

import os
import time
import argparse
import requests
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "train")
TEST_DIR  = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "SMAI-A3-MonumentApp/1.0 (academic project)"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ==========================================
# 1. Wikimedia Commons category per class
# ==========================================

WIKI_CATEGORIES = {
    "Ajanta Caves":        "Ajanta Caves",
    "Charar-E- Sharif":   "Charar-e-Sharif",
    "Chhota_Imambara":    "Chhota Imambara",
    "Ellora Caves":        "Ellora Caves",
    "Fatehpur Sikri":      "Fatehpur Sikri",
    "Gateway of India":    "Gateway of India",
    "Hawa mahal":          "Hawa Mahal",
    "Humayun_s Tomb":     "Humayun's Tomb",
    "India_gate":          "India Gate, Delhi",
    "Khajuraho":           "Khajuraho",
    "Sun Temple Konark":   "Konark Sun Temple",
    "alai_darwaza":        "Alai Darwaza",
    "alai_minar":          "Alai Minar",
    "basilica_of_bom_jesus": "Basilica of Bom Jesus",
    "charminar":           "Charminar",
    "golden temple":       "Harmandir Sahib",
    "iron_pillar":         "Iron pillar of Delhi",
    "jamali_kamali_tomb":  "Jamali Kamali mosque and tomb",
    "lotus_temple":        "Lotus Temple",
    "mysore_palace":       "Mysore Palace",
    "qutub_minar":         "Qutb Minar",
    "tajmahal":            "Taj Mahal",
    "tanjavur temple":     "Brihadeeswarar Temple",
    "victoria memorial":   "Victoria Memorial, Kolkata",
}

# ==========================================
# 2. Helpers
# ==========================================

def count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


def next_index(folder: str) -> int:
    """Return the next sequential filename index for a class folder."""
    if not os.path.isdir(folder):
        return 1
    indices = []
    for p in Path(folder).iterdir():
        try:
            indices.append(int(p.stem))
        except ValueError:
            pass
    return max(indices, default=0) + 1


def get_category_members(category: str, limit: int = 50, cont: str = None):
    """
    Returns (list of file titles, continuation token or None).
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": limit,
        "format": "json",
    }
    if cont:
        params["cmcontinue"] = cont
    try:
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=15)
        data = r.json()
        members = [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont_token = data.get("continue", {}).get("cmcontinue")
        return members, cont_token
    except Exception:
        return [], None


def get_image_url(file_title: str):
    """Return the direct download URL for a Wikimedia file title."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
    }
    try:
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=15)
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            mime = info.get("mime", "")
            url = info.get("url", "")
            width = info.get("width", 0)
            height = info.get("height", 0)
            if mime in ("image/jpeg", "image/png") and width >= 200 and height >= 200:
                return url
    except Exception:
        pass
    return None


def download_image(url: str, dst_path: str) -> bool:
    """Download a single image to dst_path. Returns True on success."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        # Reject tiny files (likely error pages or thumbnails)
        if os.path.getsize(dst_path) < 5000:
            os.remove(dst_path)
            return False
        return True
    except Exception:
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False


# ==========================================
# 3. Per-class expansion logic
# ==========================================

def expand_class(canonical: str, category: str, target: int, per_request: int):
    train_folder = os.path.join(TRAIN_DIR, canonical)
    test_folder  = os.path.join(TEST_DIR, canonical)

    current = count_images(train_folder) + count_images(test_folder)
    needed  = target - current

    if needed <= 0:
        print(f"  [{canonical}] {current} images — already at target, skipping.")
        return 0

    print(f"  [{canonical}] {current} images — downloading up to {needed} more...")
    os.makedirs(train_folder, exist_ok=True)

    downloaded = 0
    cont_token = None
    idx = next_index(train_folder)

    while downloaded < needed:
        members, cont_token = get_category_members(category, limit=per_request, cont=cont_token)
        if not members:
            break

        for title in members:
            if downloaded >= needed:
                break
            ext = os.path.splitext(title)[-1].lower()
            if ext not in IMAGE_EXTS:
                continue

            url = get_image_url(title)
            if not url:
                time.sleep(0.2)
                continue

            dst_path = os.path.join(train_folder, f"{idx}{ext}")
            if download_image(url, dst_path):
                downloaded += 1
                idx += 1
            time.sleep(0.3)

        if not cont_token:
            break  # no more pages in this category

    print(f"    → Downloaded {downloaded} images.")
    return downloaded


# ==========================================
# 4. Main
# ==========================================

def main(target: int, per_request: int):
    if not os.path.isdir(TRAIN_DIR):
        print("ERROR: Train directory not found. Run prepare_dataset.py first.")
        return

    print(f"Target images per class : {target}")
    print(f"Wikimedia batch size    : {per_request}")
    print()

    total_downloaded = 0
    for canonical, category in WIKI_CATEGORIES.items():
        n = expand_class(canonical, category, target, per_request)
        total_downloaded += n

    print(f"\nDone. Total new images downloaded: {total_downloaded}")

    # Print updated counts
    print(f"\n{'Class':<30} {'Train':>6} {'Test':>6} {'Total':>7}")
    print("-" * 52)
    for canonical in WIKI_CATEGORIES:
        tr = count_images(os.path.join(TRAIN_DIR, canonical))
        te = count_images(os.path.join(TEST_DIR, canonical))
        print(f"  {canonical:<28} {tr:>6} {te:>6} {tr+te:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, default=200,
        help="Minimum total images per class to aim for (default: 200)"
    )
    parser.add_argument(
        "--per-request", type=int, default=50,
        help="Files to request per Wikimedia API call (max 500, default: 50)"
    )
    args = parser.parse_args()
    main(args.target, args.per_request)
