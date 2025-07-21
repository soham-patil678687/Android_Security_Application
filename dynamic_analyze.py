import os
import sys
import subprocess
import time
import numpy as np
from pyaxmlparser import APK
from scapy.all import rdpcap, IP, TCP, UDP
from ipaddress import ip_address, ip_network
import joblib
import io

# Fix Windows console Unicode output issues by wrapping stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AVD_NAME    = "temp"
PCAP_DIR    = "pcap"
MODEL_PATH  = "dynamic/voting_classifier.pkl"
SCALER_PATH = "dynamic/scaler.pkl"
MONKEY_EVENTS = 600     # ~60s at 100ms throttle
# ---------------------------------------------------------------------------

# 1) Get APK path
if len(sys.argv) != 2:
    sys.exit("Usage: python dynamic_analyze.py <path_to_apk>")
apk_path = sys.argv[1]
apk_name = os.path.splitext(os.path.basename(apk_path))[0]
pcap_file = os.path.join(PCAP_DIR, f"{apk_name}_pcap.pcap")
os.makedirs(PCAP_DIR, exist_ok=True)

# 2) Extract package & main activity
apk = APK(apk_path)
pkg = apk.package
main_act = apk.get_main_activity()
if not pkg or not main_act:
    sys.exit("Error: failed to extract package or main activity.")

# 3) Launch emulator with tcpdump (suppress output)
emu_proc = subprocess.Popen([
    "emulator", "-avd", AVD_NAME,
    "-wipe-data",
    "-no-snapshot-save",
    "-tcpdump", pcap_file
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# 4) Wait for boot (suppress output)
subprocess.run(["adb", "wait-for-device"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while True:
    out = subprocess.check_output(
        ["adb", "shell", "getprop", "sys.boot_completed"],
        text=True
    ).strip()
    if out == "1":
        break
    time.sleep(1)

# 5) Install APK (suppress output)
subprocess.run(["adb", "install", "-r", apk_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 6) Launch the app (suppress output)
subprocess.run([
    "adb", "shell", "am", "start",
    "-n", f"{pkg}/{main_act}"
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 7) Run Monkey (suppress output)
monkey = subprocess.Popen([
    "adb", "shell", "monkey",
    "-p", pkg,
    "--ignore-crashes",
    "--ignore-timeouts",
    "--monitor-native-crashes",
    "--throttle", "100",
    "--pct-nav", "0",
    "--pct-majornav", "0",
    "--pct-syskeys", "0",
    "--pct-appswitch", "0",
    str(MONKEY_EVENTS)
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 8) Wait for Monkey to finish
monkey.wait()

# 9) Kill emulator (suppress output)
subprocess.run(["adb", "emu", "kill"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
emu_proc.wait()

# 10) Validate PCAP
if not os.path.exists(pcap_file):
    sys.exit(f"No PCAP found at {pcap_file}")

# 11) Parse PCAP & build features
local_nets = [
    #add local ip addresses
]

def is_local(ip):
    try:
        a = ip_address(ip)
        return any(a in net for net in local_nets)
    except:
        return False

pkts = rdpcap(pcap_file)
tcp_pkts = udp_pkts = 0
src_pkts = dst_pkts = 0
src_bytes = dst_bytes = 0
dns_count = 0
tcp_ports = set()
ext_ips = set()

for p in pkts:
    if not p.haslayer(IP):
        continue
    ip = p[IP]
    s, d = ip.src, ip.dst

    if p.haslayer(TCP):
        tcp_pkts += 1
        if p[TCP].sport == 53 or p[TCP].dport == 53:
            dns_count += 1
        if is_local(s) and not is_local(d):
            tcp_ports.add(p[TCP].dport)
        elif is_local(d) and not is_local(s):
            tcp_ports.add(p[TCP].sport)

    if p.haslayer(UDP):
        udp_pkts += 1
        if p[UDP].sport == 53 or p[UDP].dport == 53:
            dns_count += 1

    if not is_local(s):
        ext_ips.add(s)
    if not is_local(d):
        ext_ips.add(d)

    if is_local(s) and not is_local(d):
        src_pkts += 1
        src_bytes += ip.len
    elif is_local(d) and not is_local(s):
        dst_pkts += 1
        dst_bytes += ip.len

# 12) Assemble features and predict
app_pkts = src_pkts + dst_pkts
features = [
    tcp_pkts,
    len(tcp_ports),
    len(ext_ips),
    src_bytes + dst_bytes,
    udp_pkts,
    src_pkts,
    dst_pkts,
    src_bytes,
    dst_bytes,
    app_pkts,
    dns_count
]
arr = np.array([features], dtype=float)

print(arr)  # <-- your final feature vector

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
scaled = scaler.transform(arr)
pred   = model.predict(scaled)

print(f"Prediction: {pred}")
