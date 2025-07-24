import zlib
import brotli
from typing import Generator, Optional


def should_compress(filename: str, file_size: int) -> bool:
    """Kiểm tra có nên compress không"""
    if file_size < 1024 or file_size > 100 * 1024 * 1024:
        return False

    compressible = {
        ".txt",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".csv",
        ".md",
        ".log",
        ".py",
    }
    compressed = {".zip", ".rar", ".gz", ".jpg", ".png", ".pdf", ".mp4", ".mp3"}

    ext = "." + filename.lower().split(".")[-1] if "." in filename else ""
    return ext in compressible and ext not in compressed


def get_best_compression(accept_encoding: str, preferred: str) -> Optional[str]:
    """Chọn compression tốt nhất"""
    accept = accept_encoding.lower()
    if preferred == "brotli" and "br" in accept:
        return "br"
    if preferred == "gzip" and "gzip" in accept:
        return "gzip"
    if "br" in accept:
        return "br"
    if "gzip" in accept:
        return "gzip"
    return None


def compress_stream(
    stream: Generator, compression_type: str
) -> Generator[bytes, None, None]:
    if compression_type == "gzip":
        return _gzip_stream(stream)
    elif compression_type == "br":
        return _brotli_stream(stream)
    return stream


def _gzip_stream(stream: Generator) -> Generator[bytes, None, None]:
    # GZIP header (10 bytes)
    yield b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"

    compressor = zlib.compressobj(
        level=6,  # Good compression/speed balance
        method=zlib.DEFLATED,
        wbits=-15,  # Raw deflate (no zlib header)
        memLevel=8,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )

    crc = 0
    size = 0

    try:
        for chunk in stream:
            if chunk:
                crc = zlib.crc32(chunk, crc)
                size += len(chunk)
                compressed = compressor.compress(chunk)
                if compressed:
                    yield compressed

        # Flush remaining data
        final = compressor.flush()
        if final:
            yield final

        # GZIP trailer (8 bytes: CRC32 + Size)
        yield (crc & 0xFFFFFFFF).to_bytes(4, "little")
        yield (size & 0xFFFFFFFF).to_bytes(4, "little")

    except Exception:
        # Fallback: yield original data without compression
        for chunk in stream:
            if chunk:
                yield chunk


def _brotli_stream(stream: Generator) -> Generator[bytes, None, None]:
    """Brotli streaming - Tốt nhất cho text"""
    try:
        compressor = brotli.Compressor(
            quality=6,  # Good balance
            lgwin=22,  # Window size
            lgblock=0,  # Auto block size
        )

        for chunk in stream:
            if chunk:
                compressed = compressor.process(chunk)
                if compressed:
                    yield compressed

        # Finish compression
        final = compressor.finish()
        if final:
            yield final

    except Exception:
        # Fallback: yield original data
        for chunk in stream:
            if chunk:
                yield chunk
