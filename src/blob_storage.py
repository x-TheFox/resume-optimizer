"""
BlobStorage – Handles file storage via Vercel Blob or local filesystem fallback.

In production (Vercel), files are stored in Vercel Blob and served via public URLs.
In development (local), files are stored on disk as before.
"""

import os
import io
import json
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Vercel Blob API endpoint
BLOB_API_URL = "https://blob.vercel-storage.com"


class BlobStorage:
    """Abstraction layer for file storage — Vercel Blob in prod, local in dev."""

    def __init__(self):
        self.token = os.getenv("BLOB_READ_WRITE_TOKEN", "")
        self.is_vercel = bool(self.token) and bool(os.getenv("VERCEL", ""))
        if self.is_vercel:
            logger.info("BlobStorage: Using Vercel Blob")
        else:
            logger.info("BlobStorage: Using local filesystem fallback")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def upload_file(self, file_bytes: bytes, filename: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload a file and return its URL (Blob URL or local path).

        Args:
            file_bytes: Raw bytes of the file
            filename: Desired filename (will get random suffix on Blob)
            content_type: MIME type

        Returns:
            URL string for downloading the file
        """
        if self.is_vercel:
            return self._upload_to_blob(file_bytes, filename, content_type)
        else:
            return self._save_locally(file_bytes, filename)

    def upload_resume(self, file_stream, filename: str) -> tuple:
        """
        Upload a resume .docx file.

        Args:
            file_stream: Werkzeug FileStorage object
            filename: Sanitized filename

        Returns:
            Tuple of (url_or_path, file_bytes) — bytes are needed for processing
        """
        file_bytes = file_stream.read()
        url = self.upload_file(file_bytes, filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        return url, file_bytes

    def save_pdf(self, pdf_buffer: io.BytesIO, filename: str) -> str:
        """
        Save a PDF from a BytesIO buffer and return its URL.

        Args:
            pdf_buffer: BytesIO containing PDF data
            filename: Desired filename

        Returns:
            URL string for downloading the PDF
        """
        pdf_bytes = pdf_buffer.getvalue()
        return self.upload_file(pdf_bytes, filename, "application/pdf")

    def save_docx(self, docx_buffer: io.BytesIO, filename: str) -> str:
        """
        Save a .docx from a BytesIO buffer and return its URL.

        Args:
            docx_buffer: BytesIO containing .docx data
            filename: Desired filename

        Returns:
            URL string for downloading the .docx
        """
        docx_bytes = docx_buffer.getvalue()
        return self.upload_file(
            docx_bytes, filename,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def get_download_url(self, url_or_path: str) -> str:
        """
        Get a download-ready URL for a stored file.

        For Vercel Blob: returns the blob URL directly (publicly accessible).
        For local: returns the /download/<filename> path.
        """
        if self.is_vercel:
            # Blob URLs are already publicly accessible
            return url_or_path
        else:
            # Local: extract filename from path and return download route
            filename = os.path.basename(url_or_path)
            return f"/download/{filename}"

    def cleanup_old_files(self, max_age_hours: int = 24) -> dict:
        """
        Delete files older than max_age_hours.

        For Vercel Blob: lists and deletes old blobs via API.
        For local: deletes old files from outputs/ directory.

        Returns:
            Dict with cleanup stats
        """
        if self.is_vercel:
            return self._cleanup_blob(max_age_hours)
        else:
            return self._cleanup_local(max_age_hours)

    # ------------------------------------------------------------------ #
    #  Vercel Blob Operations
    # ------------------------------------------------------------------ #

    def _upload_to_blob(self, file_bytes: bytes, filename: str, content_type: str) -> str:
        """Upload file to Vercel Blob and return the public URL."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": content_type,
                    "x-api-version": "7",
                }

                # Use the put API
                response = requests.put(
                    f"{BLOB_API_URL}/{filename}",
                    headers=headers,
                    data=file_bytes,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                url = result.get("url", "")
                logger.info("Uploaded to Blob: %s -> %s", filename, url)
                return url

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", 0)
                if status in (502, 503, 504, 429) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    logger.warning("Blob upload %d/%d got %s, retrying in %ds...", attempt + 1, max_retries, status, wait)
                    import time; time.sleep(wait)
                    continue
                logger.error("Blob upload failed: %s", e)
                raise RuntimeError(f"Failed to upload file to storage: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Blob upload %d/%d failed (%s), retrying in %ds...", attempt + 1, max_retries, e, wait)
                    import time; time.sleep(wait)
                    continue
                logger.error("Blob upload failed: %s", e)
                raise RuntimeError(f"Failed to upload file to storage: {e}")

    def _cleanup_blob(self, max_age_hours: int) -> dict:
        """List and delete old blobs."""
        deleted = 0
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "x-api-version": "7",
            }
            response = requests.get(
                f"{BLOB_API_URL}",
                headers=headers,
                params={"limit": 100},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            cutoff = time.time() - (max_age_hours * 3600)
            urls_to_delete = []

            for blob in data.get("blobs", []):
                uploaded_at = blob.get("uploadedAt", "")
                if uploaded_at:
                    # Parse ISO timestamp
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
                        if dt.timestamp() < cutoff:
                            urls_to_delete.append(blob["url"])
                    except (ValueError, KeyError):
                        pass

            # Delete old blobs
            if urls_to_delete:
                delete_response = requests.post(
                    f"{BLOB_API_URL}/delete",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        "x-api-version": "7",
                    },
                    json={"urls": urls_to_delete},
                    timeout=15,
                )
                delete_response.raise_for_status()
                deleted = len(urls_to_delete)

        except Exception as e:
            logger.error("Blob cleanup failed: %s", e)

        return {"deleted": deleted}

    # ------------------------------------------------------------------ #
    #  Local Filesystem Operations (Development Fallback)
    # ------------------------------------------------------------------ #

    def _save_locally(self, file_bytes: bytes, filename: str) -> str:
        """Save file to local outputs/ directory."""
        from config import Config
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        file_path = os.path.join(Config.OUTPUT_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info("Saved locally: %s", file_path)
        return file_path

    def _cleanup_local(self, max_age_hours: int) -> dict:
        """Delete old files from outputs/ directory."""
        from config import Config
        deleted = 0
        cutoff = time.time() - (max_age_hours * 3600)

        for folder in [Config.OUTPUT_FOLDER, Config.UPLOAD_FOLDER]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                    try:
                        os.remove(filepath)
                        deleted += 1
                    except OSError:
                        pass

        return {"deleted": deleted}
