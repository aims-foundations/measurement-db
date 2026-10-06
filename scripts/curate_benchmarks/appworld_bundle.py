"""Read provider-distributed AppWorld bundles using its published file format.

The format constants are public values shipped with AppWorld, not credentials.
See https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/common/utils.py.
"""

import io
import zipfile

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    from cryptography.hazmat.decrepit.ciphers.modes import CFB
except ImportError:
    from cryptography.hazmat.primitives.ciphers.modes import CFB


PASSWORD = "WEquKLy##9M@qu"
SALT = b"Nvx#rYcYQ2%btf"


def open_bundle(path):
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=SALT, iterations=100000
    ).derive(PASSWORD.encode())
    data = path.read_bytes()
    decryptor = Cipher(algorithms.AES(key), CFB(data[:16])).decryptor()
    return zipfile.ZipFile(
        io.BytesIO(decryptor.update(data[16:]) + decryptor.finalize())
    )
