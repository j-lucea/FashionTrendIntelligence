"""HTTP proxy for Hugging Face clothes segmentation."""

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


def reply(status, payload, headers=None):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **(headers or {})},
        "body": json.dumps(payload, separators=(",", ":")),
    }


def handler(event, context):
    if event["rawPath"] != "/segment":
        return reply(404, {"error": "Unknown route"})
    if event["requestContext"]["http"]["method"] != "POST":
        return reply(405, {"error": "Use POST"}, {"allow": "POST"})

    headers = {name.lower(): value for name, value in event.get("headers", {}).items()}
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in IMAGE_TYPES:
        return reply(415, {"error": "Use image/jpeg, image/png or image/webp"})

    body = event.get("body")
    if not body or not event.get("isBase64Encoded"):
        return reply(400, {"error": "Send raw image bytes in the request body"})
    if len(body) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        return reply(413, {"error": "Image exceeds 4 MiB"})
    try:
        image = base64.b64decode(body, validate=True)
    except ValueError:
        return reply(400, {"error": "Invalid image payload"})
    if len(image) > MAX_IMAGE_BYTES:
        return reply(413, {"error": "Image exceeds 4 MiB"})

    token = os.environ.get("HF_TOKEN")
    if not token:
        return reply(500, {"error": "Inference token is missing"})
    request = Request(
        MODEL_URL,
        data=image,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        if error.code in {400, 413, 415, 422}:
            return reply(error.code, {"error": "Hugging Face rejected the image"})
        if error.code in {429, 503}:
            retry_after = error.headers.get("Retry-After")
            headers = {"retry-after": retry_after} if retry_after else None
            return reply(error.code, {"error": "Inference temporarily unavailable"}, headers)
        return reply(502, {"error": "Hugging Face request failed", "upstream_status": error.code})
    except TimeoutError:
        return reply(504, {"error": "Hugging Face timed out"})
    except (URLError, OSError):
        return reply(502, {"error": "Hugging Face connection failed"})

    if len(data) > MAX_RESPONSE_BYTES:
        return reply(502, {"error": "Segmentation response exceeds 5 MiB"})
    try:
        masks = json.loads(data)
    except ValueError:
        return reply(502, {"error": "Invalid Hugging Face response"})
    if not isinstance(masks, list) or not all(
        isinstance(mask, dict)
        and isinstance(mask.get("label"), str)
        and isinstance(mask.get("mask"), str)
        for mask in masks
    ):
        return reply(502, {"error": "Invalid segmentation response"})
    return reply(200, masks)
