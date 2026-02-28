"""
Safari cookie extraction for Twitter/X authentication.

Parses the macOS Safari binary cookies file to extract
auth_token and ct0 cookies for x.com.
"""

import struct

__all__ = ["get_safari_cookies"]

SAFARI_COOKIE_PATH = "{home}/Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies"


def get_safari_cookies(
    home: str = "",
) -> dict[str, str]:
    """
    Extract auth_token and ct0 cookies for x.com from Safari.

    Parameters
    ----------
    home : str, optional
        Home directory path. Defaults to ``~`` via path expansion.

    Returns
    -------
    dict[str, str]
        Dictionary with ``auth_token`` and ``ct0`` keys if found.

    Raises
    ------
    FileNotFoundError
        If the Safari cookie file does not exist.
    RuntimeError
        If the required cookies are not found.
    """
    import os

    if not home:
        home = os.path.expanduser("~")

    path = SAFARI_COOKIE_PATH.format(home=home)

    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != b"cook":
        raise RuntimeError(f"Not a Safari binarycookies file: {path}")

    num_pages = struct.unpack(">I", data[4:8])[0]
    offset = 8

    page_sizes = []
    for _ in range(num_pages):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        page_sizes.append(size)
        offset += 4

    result: dict[str, str] = {}

    for page_size in page_sizes:
        page_data = data[offset : offset + page_size]
        offset += page_size

        if len(page_data) < 8:
            continue

        num_cookies = struct.unpack("<I", page_data[4:8])[0]
        cookie_offsets = [struct.unpack("<I", page_data[8 + i * 4 : 12 + i * 4])[0] for i in range(num_cookies)]

        for co in cookie_offsets:
            cookie_data = page_data[co:]
            if len(cookie_data) < 44:
                continue

            url_off = struct.unpack("<I", cookie_data[16:20])[0]
            name_off = struct.unpack("<I", cookie_data[20:24])[0]
            val_off = struct.unpack("<I", cookie_data[28:32])[0]

            try:
                domain = cookie_data[url_off : cookie_data.index(b"\x00", url_off)].decode()
                name = cookie_data[name_off : cookie_data.index(b"\x00", name_off)].decode()
                value = cookie_data[val_off : cookie_data.index(b"\x00", val_off)].decode()
            except (ValueError, UnicodeDecodeError):
                continue

            if "x.com" not in domain and "twitter.com" not in domain:
                continue

            if name in ("auth_token", "ct0"):
                result[name] = value

    if "auth_token" not in result or "ct0" not in result:
        raise RuntimeError(
            "Could not find auth_token and ct0 cookies for x.com in Safari. "
            "Make sure you are logged into x.com in Safari."
        )

    return result
