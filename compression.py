import struct

# -------- .cmpt365 file --------
# magic[8]        = b"CMPT365\0"
# width           u32
# height          u32
# original_size   u32
# compressed_size u32
# payload: encoded bytes
CMPT365_MAGIC = b"CMPT365\x00"
HEADER_STRUCT = "<8sIIII"

def pack_header(width, height, original_size, compressed_size):
    return struct.pack(
        HEADER_STRUCT,
        CMPT365_MAGIC,
        width,
        height,
        original_size,
        compressed_size,
    )

def unpack_header(data: bytes):
    magic, width, height, original_size, compressed_size = struct.unpack(
        HEADER_STRUCT, data[:struct.calcsize(HEADER_STRUCT)]
    )
    return {
        "magic": magic,
        "width": width,
        "height": height,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }

MAX_BITS = 16
MAX_DICT = (1 << MAX_BITS)

# Encode function adapted and ported to Python from GeeksForGeeks (LZW) and CMPT 365 course slides.
# https://www.geeksforgeeks.org/computer-networks/lzw-lempel-ziv-welch-compression-technique/
def lzw_encode(src_bytes: bytes) -> bytes:
    if not src_bytes:
        return bytes()

    table = {bytes([i]): i for i in range(256)}
    p = src_bytes[:1]
    next_code = 256
    out: list[int] = []

    for b in src_bytes[1:]:
        c = bytes([b])
        pc = p + c
        if pc in table:
            p = pc
        else:
            out.append(table[p])
            if next_code < MAX_DICT:
                table[pc] = next_code
                next_code += 1
            p = c

    out.append(table[p])
    encoded = struct.pack("<" + "H" * len(out), *out)
    return bytes(encoded)

# Decode function adapted and ported to Python from GeeksForGeeks (LZW) and CMPT 365 course slides.
# https://www.geeksforgeeks.org/computer-networks/lzw-lempel-ziv-welch-compression-technique/
def lzw_decode(src_encoded: bytes) -> bytes:
    if not src_encoded:
        return b""

    num_codes = len(src_encoded) // 2
    codes = list(struct.unpack("<" + "H" * num_codes, src_encoded))

    table = {i: bytes([i]) for i in range(256)}
    next_code = 256

    old = codes[0]
    s = table[old]
    out = bytearray(s)
    c = s[:1]

    for n in codes[1:]:
        if n in table:
            s = table[n]
        else:
            s = table[old] + c

        out.extend(s)
        c = s[:1]

        if next_code < MAX_DICT:
            table[next_code] = table[old] + c
            next_code += 1

        old = n

    return bytes(out)