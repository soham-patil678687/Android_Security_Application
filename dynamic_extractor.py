import os
import sys
import subprocess
import time
import shutil
import numpy as np
from pyaxmlparser import APK
from scapy.all import rdpcap, IP, TCP, UDP
from ipaddress import ip_address, ip_network
import joblib

# 1) Check that tshark is installed
if shutil.which("tshark") is None:
    sys.exit("Error: *tshark* not found in PATH. Please install Wireshark CLI.")

# 2) Get the APK path from the command line
if len(sys.argv) != 2:
    sys.exit("Usage: python script.py <path_to_apk>")
apk_path = sys.argv[1]
apk_name = os.path.splitext(os.path.basename(apk_path))[0]

# 3) Extract package name & main activity from the APK
apk = APK(apk_path)
pkg = apk.package
main_act = apk.get_main_activity()
if not pkg or not main_act:
    sys.exit("Error: failed to extract package or main activity.")

# 4) Ensure 'pcap' directory exists
pcap_dir = "pcap"
os.makedirs(pcap_dir, exist_ok=True)
pcap_file = os.path.join(pcap_dir, f"{apk_name}_pcap.pcap")

# 5) Auto-detect TShark interface (first non-loopback)
def find_tshark_iface():
    out = subprocess.check_output(["tshark", "-D"], text=True)
    for line in out.splitlines():
        idx, name = line.split(".", 1)
        name = name.strip()
        if "Loopback" in name or "lo" in name.lower():
            return idx.strip()  # Return the loopback interface
    sys.exit("Error: no suitable TShark interface found.")



iface = find_tshark_iface()

# 6) Start (or attach to) the AVD named "temp"
emulator_serial = None

# Check if any emulator is already running:
adb_devices = subprocess.check_output(["adb", "devices"], text=True)
device_lines = [line for line in adb_devices.strip().splitlines()[1:] if line.strip()]

for line in device_lines:
    serial, state = line.split()
    if serial.startswith("emulator-"):
        emulator_serial = serial
        break

if emulator_serial is None:
    print("[+] Launching emulator 'temp' …")
    subprocess.Popen(
        ["emulator", "-avd", "temp", "-read-only"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    subprocess.run(["adb", "wait-for-device"], check=True)
    booted = ""
    while "1" not in booted.strip():
        time.sleep(2)
        booted = subprocess.check_output(
            ["adb", "shell", "getprop", "sys.boot_completed"], text=True
        )
    # Re-fetch the serial now that it's online
    adb_devices = subprocess.check_output(["adb", "devices"], text=True)
    device_lines = [line for line in adb_devices.strip().splitlines()[1:] if line.strip()]
    for line in device_lines:
        serial, state = line.split()
        if serial.startswith("emulator-"):
            emulator_serial = serial
            break
    print(f"[+] Emulator launched and running as {emulator_serial}.")
else:
    print(f"[+] Found existing emulator: {emulator_serial}")

# ✅ Safe to fetch emulator network info now
print("[+] Emulator network interfaces:")
print(subprocess.check_output(["adb", "-s", emulator_serial, "shell", "ip", "addr", "show"], text=True))

# 7) Install the APK on the emulator/device
print(f"[+] Installing {apk_name}.apk → emulator")
subprocess.run(["adb", "-s", emulator_serial, "install", "-r", apk_path], check=True)

# 8) Launch the app via Monkey (simulate for ~60s)
print("[+] Simulating user interaction (Monkey) for 60s …")
monkey = subprocess.Popen([
    "adb", "-s", emulator_serial, "shell", "monkey",
    "-p", pkg,
    "--ignore-crashes",
    "--ignore-timeouts",
    "--monitor-native-crashes",
    "--throttle", "100",
    "--pct-nav", "0",
    "--pct-majornav", "0",
    "--pct-syskeys", "0",
    "--pct-appswitch", "0",
    "600"
])

# 9) In parallel, run TShark to capture packets
print(f"[+] Capturing traffic on iface {iface} to {pcap_file} …")
tshark = subprocess.Popen([
    "tshark", "-i", iface,
    "-a", "duration:60",
    "-w", pcap_file
])
tshark.wait()

# 10) Stop Monkey if still running
monkey.terminate()

# 11) Parse the PCAP with Scapy to compute features
local_nets = [
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8")
]
def is_local(ip):
    a = ip_address(ip)
    return any(a in net for net in local_nets)

pkts = rdpcap(pcap_file)
tcp_pkts = udp_pkts = 0
src_pkts = dst_pkts = 0
src_bytes = dst_bytes = 0
dns_count = 0
tcp_ports = set()
ext_ips = set()
times = []

for p in pkts:
    if not p.haslayer(IP):
        continue
    ip = p[IP]
    times.append(p.time)
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

# 12) Assemble selected features
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

# 13) Convert to a NumPy array and print
arr = np.array([features], dtype=float)
print("\nExtracted feature array:")
print(arr)

# 14) Load scaler and model, then scale and predict
model = joblib.load('dynamic/voting_classifier.pkl')
scaler = joblib.load('dynamic/scaler.pkl')
scaled_arr = scaler.transform(arr)
prediction = model.predict(scaled_arr)
print(f"[+] Prediction: {prediction}")

print(f"[+] DNS Query Count: {dns_count}")
print("[+] Analysis done. Cleaning up…")

# 15) Kill the emulator via ADB to close its GUI window
subprocess.run(["adb", "-s", emulator_serial, "emu", "kill"], check=True)
print(f"[+] Emulator '{emulator_serial}' terminated via ADB.")
