"""Segment clothing in images with the Hugging Face Inference API."""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

MODEL_ID = "sayeed99/segformer_b3_clothes"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
CLASS_NAMES = (
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(inputs: Sequence[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of supported images."""
    images: list[Path] = []
    for path in inputs:
        if not path.exists():
            raise ValueError(f"Chemin introuvable : {path}")
        if path.is_dir():
            images.extend(
                image
                for image in sorted(path.iterdir())
                if image.is_file() and image.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
        else:
            raise ValueError(f"Format d'image non pris en charge : {path}")

    if not images:
        raise ValueError("Aucune image JPG, PNG ou WEBP trouvée.")
    return images


def request_segmentation(
    image_path: Path, token: str
) -> list[dict[str, object]]:
    """Send an image to Hugging Face and return its segmentation results."""
    content_type = "image/png" if image_path.suffix.lower() == ".png" else (
        "image/webp" if image_path.suffix.lower() == ".webp" else "image/jpeg"
    )
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        data=image_path.read_bytes(),
        timeout=120,
    )
    if not response.ok:
        details = response.text[:500]
        raise RuntimeError(
            f"Erreur de l'API Hugging Face ({response.status_code}) : {details}"
        )

    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"Erreur de l'API Hugging Face : {payload['error']}")
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("Format de réponse inattendu de l'API Hugging Face.")
    return payload


def decode_mask(encoded_mask: str, size: tuple[int, int]) -> np.ndarray:
    """Decode a base64 mask and resize it to the original image dimensions."""
    mask_bytes = base64.b64decode(encoded_mask, validate=True)
    with Image.open(io.BytesIO(mask_bytes)) as mask_image:
        resized = mask_image.convert("L").resize(size, Image.Resampling.NEAREST)
        return np.asarray(resized)


def combine_masks(
    results: Sequence[Mapping[str, object]], size: tuple[int, int]
) -> np.ndarray:
    """Combine the API's per-class masks into one class-index image."""
    width, height = size
    combined = np.zeros((height, width), dtype=np.uint8)
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}

    for result in results:
        label = result.get("label")
        encoded_mask = result.get("mask")
        if not isinstance(label, str) or not isinstance(encoded_mask, str):
            raise ValueError("Un masque reçu de l'API ne contient pas label et mask.")
        if label == "Background":
            continue
        if label not in class_ids:
            raise ValueError(f"Classe inconnue dans la réponse de l'API : {label}")
        mask = decode_mask(encoded_mask, size)
        combined[mask > 0] = class_ids[label]

    return combined


def display_results(image_paths: Sequence[Path], token: str) -> None:
    """Fetch and display the original images, class masks, and overlays."""
    palette_colors = list(plt.get_cmap("tab20").colors[: len(CLASS_NAMES)])
    palette_colors[0] = (0.12, 0.12, 0.12)
    palette = ListedColormap(palette_colors)

    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_array = np.asarray(image)
            mask = combine_masks(request_segmentation(image_path, token), image.size)

        figure, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
        axes[0].imshow(image_array)
        axes[0].set_title("Image originale")
        axes[1].imshow(mask, cmap=palette, vmin=0, vmax=len(CLASS_NAMES) - 1)
        axes[1].set_title("Segmentation")
        axes[2].imshow(image_array)
        overlay_palette = ListedColormap(palette_colors)
        overlay_palette.set_bad(alpha=0)
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[2].imshow(
            overlay,
            cmap=overlay_palette,
            vmin=0,
            vmax=len(CLASS_NAMES) - 1,
            alpha=0.55,
        )
        axes[2].set_title("Superposition")

        detected_ids = np.unique(mask)
        legend = [
            Patch(
                facecolor=palette(class_id),
                label=CLASS_NAMES[class_id],
            )
            for class_id in detected_ids
            if class_id != 0
        ]
        if legend:
            figure.legend(handles=legend, loc="outside lower center", ncol=6)
        for axis in axes:
            axis.axis("off")
        figure.suptitle(image_path.name)
        plt.show()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Segmentez des vêtements avec l'API Hugging Face."
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Fichiers image ou dossiers contenant des images JPG, PNG ou WEBP.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error(
            "Définissez le token Hugging Face dans la variable d'environnement HF_TOKEN "
            "ou dans un fichier .env."
        )

    try:
        image_paths = collect_images(args.images)
        display_results(image_paths, token)
    except (OSError, requests.RequestException, RuntimeError, ValueError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
