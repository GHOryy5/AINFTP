import socket
import struct
import sys

INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"
MAGIC = 0x4E465450


sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
sock.bind((INTERFACE, 0))

# Ethernet Header (Broadcast)
eth_header = struct.pack("!6s6sH", b'\xff\xff\xff\xff\xff\xff', b'\x00\x00\x00\x00\x00\x00', 0x0800)
# IP Header (Empty)
ip_header = b'\x00' * 20
# NFTP Header [Magic, Ver, Op, Node, Rank, Len, Sig]
nftp_header = struct.pack("!IBBHIIQ", MAGIC, 1, 1, 77, 0, 0, 0)

sock.send(eth_header + ip_header + nftp_header)
print(f"[*] IMPULSE SENT. Magic: {hex(MAGIC)}")
