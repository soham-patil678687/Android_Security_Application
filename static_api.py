from flask import Flask, request, jsonify
from androguard.misc import AnalyzeAPK
import os
import re
import zipfile
import tempfile
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import Counter
import pickle
import pandas as pd
from urllib.parse import urlparse
import ipaddress
app = Flask(__name__)



# ──────── LOAD ONCE ──────── #
print("[*] Loading scaler and models…")
apk_scaler   = joblib.load("scaler_cnn_bilstm_initial.pkl")
rf       = joblib.load("Random_Forest.pkl")
et       = joblib.load("Extra_Trees.pkl")
gb       = joblib.load("Gradient_Boosting_Machine.pkl")
ab       = joblib.load("AdaBoost.pkl")
dl_model = load_model("model_bilstm_cnn_best.keras")

SCALER_PATH = 'url_scaler.pkl'
MODEL_PATH = 'url.pkl'


print("[✓] Scaler and models loaded.")


with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ──────── CONFIG (FEATURE LISTS & LABELS) ──────── #
ALL_FEATURES = [
    'ACCESS_PERSONAL_INFO___', 'CREATE_FOLDER_____', 'CREATE_PROCESS_____', 'CREATE_THREAD_____', 'DEVICE_ACCESS_____',
    'EXECUTE_____', 'FS_ACCESS____', 'FS_ACCESS()____', 'FS_ACCESS(CREATE)____', 'FS_ACCESS(CREATE__APPEND)__',
    'FS_ACCESS(CREATE__READ)__', 'FS_ACCESS(CREATE__READ__WRITE)', 'FS_ACCESS(CREATE__WRITE)__',
    'FS_ACCESS(CREATE__WRITE__APPEND)', 'FS_ACCESS(READ)____', 'FS_ACCESS(WRITE)____', 'FS_PIPE_ACCESS___',
    'FS_PIPE_ACCESS(READ)___', 'FS_PIPE_ACCESS(READ__)_', 'FS_PIPE_ACCESS(READ__WRITE)_', 'FS_PIPE_ACCESS(WRITE)___',
    'NETWORK_ACCESS____', 'NETWORK_ACCESS()____', 'NETWORK_ACCESS(READ__WRITE)__', 'NETWORK_ACCESS(WRITE)____',
    'NETWORK_ACCESS(WRITE__)__', 'SMS_SEND____', 'TERMINATE_PROCESS', 'TERMINATE_THREAD', '__arm_nr_cacheflush',
    '__arm_nr_set_tls', '_llseek', '_newselect', 'access', 'addAccessibilityInteractionConnection', 'addClient',
    'addToDisplay', 'addToDisplayWithoutInputChannel', 'bind', 'brk', 'cancelAllNotifications',
    'cancelNotificationWithTag', 'chdir', 'checkOperation', 'checkPackage', 'checkPermission', 'chmod',
    'clock_gettime', 'clone', 'close', 'connect', 'dup', 'dup2', 'enqueueNotificationWithTag', 'enqueueToast',
    'epoll_create', 'epoll_ctl', 'epoll_wait', 'execve', 'exit', 'exit_group', 'fchmod', 'fchown32', 'fcntl64',
    'fdatasync', 'finishDrawing', 'finishInput', 'flock', 'fork', 'fstat64', 'fsync', 'ftruncate', 'ftruncate64',
    'futex', 'getAccounts', 'getAccountsAsUser', 'getActiveNetworkInfo', 'getActivePhoneType', 'getActivityInfo',
    'getAllNetworkInfo', 'getAnimationScale', 'getApplicationInfo', 'getApplicationRestrictions', 'getBestProvider',
    'getCallState', 'getCameraInfo', 'getCellLocation', 'getComponentEnabledSetting', 'getConnectionInfo',
    'getCurrentSpellChecker', 'getCurrentSpellCheckerSubtype', 'getDataNetworkType', 'getDeviceId', 'getDeviceSvn',
    'getDisplayInfo', 'getEnabledAccessibilityServiceList', 'getIccSerialNumber', 'getInTouchMode', 'getInputDevice',
    'getInputDeviceIds', 'getInstalledApplications', 'getInstalledPackages', 'getInstallerPackageName',
    'getLastLocation', 'getLine1Number', 'getMode', 'getNetworkInfo', 'getNightMode', 'getNumberOfCameras',
    'getPackageInfo', 'getPackagesForUid', 'getReceiverInfo', 'getRingerMode', 'getSearchableInfo', 'getServiceInfo',
    'getSpellCheckerService', 'getStreamMaxVolume', 'getStreamVolume', 'getSubscriberId', 'getUsers',
    'getVoiceMailNumber', 'getWifiEnabledState', 'getWifiServiceMessenger', 'getcwd', 'getdents64', 'getegid32',
    'geteuid32', 'getgid32', 'getpid', 'getppid', 'getpriority', 'getsockname', 'getsockopt', 'gettid', 'gettimeofday',
    'getuid32', 'hasNavigationBar', 'hasSystemFeature', 'inKeyguardRestrictedInputMode', 'inotify_add_watch',
    'inotify_init', 'ioctl', 'isActiveNetworkMetered', 'isAdminActive', 'isCameraSoundForced', 'isImsSmsSupported',
    'isProviderEnabled', 'isScreenOn', 'isSpeakerphoneOn', 'isSpellCheckerEnabled', 'listen', 'locationCallbackFinished',
    'lseek', 'lstat64', 'madvise', 'mkdir', 'mmap2', 'mprotect', 'msync', 'munmap', 'nanosleep', 'notifyChange',
    'onGetSentenceSuggestionsMultiple', 'onRectangleOnScreenRequested', 'open', 'openSession', 'performDeferredDestroy',
    'pipe', 'poll', 'prctl', 'pread64', 'pwrite64', 'queryIntentActivities', 'queryIntentReceivers', 'queryIntentServices',
    'read', 'readlink', 'recvfrom', 'recvmsg', 'registerContentObserver', 'registerInputDevicesChangedListener', 'relayout',
    'releaseWakeLock', 'releaseWifiLock', 'remove', 'rename', 'requestLocationUpdates', 'resolveContentProvider',
    'resolveIntent', 'rmdir', 'sched_getparam', 'sched_getscheduler', 'sched_yield', 'send', 'sendAccessibilityEvent',
    'sendText', 'sendmsg', 'sendto', 'set', 'setComponentEnabledSetting', 'setInTouchMode', 'setMobileDataEnabled',
    'setMode', 'setRingerMode', 'setTextAfterCursor', 'setTextBeforeCursor', 'setTransparentRegion', 'setWifiEnabled',
    'setpgid', 'setpriority', 'setresgid32', 'setresuid32', 'setsockopt', 'shutdown', 'sigaction', 'sigaltstack',
    'sigprocmask', 'socket', 'socketpair', 'startInput', 'startScan', 'stat64', 'statfs64', 'statusBarVisibilityChanged',
    'ugetrlimit', 'uname', 'unlink', 'unregisterContentObserver', 'updateSelection', 'utimes', 'vfork', 'wait4',
    'windowGainedFocus', 'write', 'writev'
]

STATIC_FEATURES = [
    'ACCESS_PERSONAL_INFO___', 'CREATE_FOLDER_____', 'DEVICE_ACCESS_____',
    'EXECUTE_____', 'FS_ACCESS____', 'FS_ACCESS()____', 'FS_ACCESS(CREATE)____',
    'FS_ACCESS(CREATE__APPEND)__', 'FS_ACCESS(READ)____', 'FS_ACCESS(WRITE)____',
    'FS_PIPE_ACCESS___', 'FS_PIPE_ACCESS(READ)___', 'FS_PIPE_ACCESS(READ__)_',
    'FS_PIPE_ACCESS(WRITE)___', 'NETWORK_ACCESS____', 'NETWORK_ACCESS()____',
    'NETWORK_ACCESS(READ__WRITE)__', 'NETWORK_ACCESS(WRITE)____',
    'NETWORK_ACCESS(WRITE__)__', 'SMS_SEND____', 'checkOperation',
    'checkPackage', 'checkPermission', 'getAccounts', 'getAccountsAsUser',
    'getActiveNetworkInfo', 'getActivePhoneType', 'getActivityInfo',
    'getAllNetworkInfo', 'getAnimationScale', 'getApplicationInfo',
    'getApplicationRestrictions', 'getBestProvider', 'getCallState',
    'getCameraInfo', 'getCellLocation', 'getComponentEnabledSetting',
    'getConnectionInfo', 'getCurrentSpellChecker', 'getCurrentSpellCheckerSubtype',
    'getDataNetworkType', 'getDeviceId', 'getDeviceSvn', 'getDisplayInfo',
    'getEnabledAccessibilityServiceList', 'getIccSerialNumber',
    'getInTouchMode', 'getInputDevice', 'getInputDeviceIds',
    'getInstalledApplications', 'getInstalledPackages', 'getInstallerPackageName',
    'getLastLocation', 'getLine1Number', 'getMode', 'getNetworkInfo',
    'getNightMode', 'getNumberOfCameras', 'getPackageInfo', 'getPackagesForUid',
    'getReceiverInfo', 'getRingerMode', 'getSearchableInfo', 'getServiceInfo',
    'getSpellCheckerService', 'getStreamMaxVolume', 'getStreamVolume',
    'getSubscriberId', 'getUsers', 'getVoiceMailNumber', 'getWifiEnabledState',
    'getWifiServiceMessenger', 'hasNavigationBar', 'hasSystemFeature',
    'inKeyguardRestrictedInputMode', 'isActiveNetworkMetered', 'isAdminActive',
    'isCameraSoundForced', 'isImsSmsSupported', 'isProviderEnabled',
    'isScreenOn', 'isSpeakerphoneOn', 'isSpellCheckerEnabled',
    'locationCallbackFinished', 'notifyChange', 'onGetSentenceSuggestionsMultiple',
    'onRectangleOnScreenRequested', 'performDeferredDestroy', 'queryIntentActivities',
    'queryIntentReceivers', 'queryIntentServices', 'registerContentObserver',
    'registerInputDevicesChangedListener', 'relayout', 'releaseWakeLock',
    'releaseWifiLock', 'requestLocationUpdates', 'resolveContentProvider',
    'resolveIntent', 'sendAccessibilityEvent', 'sendText', 'set',
    'setComponentEnabledSetting', 'setInTouchMode', 'setMobileDataEnabled',
    'setMode', 'setRingerMode', 'setTextAfterCursor', 'setTextBeforeCursor',
    'setTransparentRegion', 'setWifiEnabled', 'startInput', 'startScan',
    'statusBarVisibilityChanged', 'unregisterContentObserver', 'updateSelection',
    'windowGainedFocus', 'Class', 'openSession', 'addClient',
    'enqueueNotificationWithTag', 'cancelNotificationWithTag', 'enqueueToast',
    'cancelAllNotifications', 'addAccessibilityInteractionConnection', 'addToDisplay',
    'addToDisplayWithoutInputChannel'
]

PSEUDO_FEATURES = [
    'CREATE_PROCESS_____', 'CREATE_THREAD_____', 'TERMINATE_PROCESS',
    'TERMINATE_THREAD', '__arm_nr_cacheflush', '__arm_nr_set_tls', '_llseek',
    '_newselect', 'access', 'bind', 'brk', 'chdir', 'chmod', 'clock_gettime',
    'clone', 'close', 'connect', 'dup', 'dup2', 'epoll_create', 'epoll_ctl',
    'epoll_wait', 'execve', 'exit', 'exit_group', 'fchmod', 'fchown32',
    'fcntl64', 'fdatasync', 'finishDrawing', 'finishInput', 'flock', 'fork',
    'fstat64', 'fsync', 'ftruncate', 'ftruncate64', 'futex', 'getcwd',
    'getdents64', 'getegid32', 'geteuid32', 'getgid32', 'getpid', 'getppid',
    'getpriority', 'getsockname', 'getsockopt', 'gettid', 'gettimeofday',
    'getuid32', 'inotify_add_watch', 'inotify_init', 'ioctl', 'listen', 'lseek',
    'lstat64', 'madvise', 'mkdir', 'mmap2', 'mprotect', 'msync', 'munmap',
    'nanosleep', 'open', 'pipe', 'poll', 'prctl', 'pread64', 'pwrite64',
    'read', 'readlink', 'recvfrom', 'recvmsg', 'remove', 'rename', 'rmdir',
    'sched_getparam', 'sched_getscheduler', 'sched_yield', 'send', 'sendmsg',
    'sendto', 'setpgid', 'setpriority', 'setresgid32', 'setresuid32',
    'setsockopt', 'shutdown', 'sigaction', 'sigaltstack', 'sigprocmask',
    'socket', 'socketpair', 'stat64', 'statfs64', 'ugetrlimit', 'uname',
    'unlink', 'utimes', 'vfork', 'wait4', 'write', 'writev'
]

LABEL_MAP = [
    "Adware",       # 0
    "Banking",      # 1
    "SMS malware",  # 2
    "Riskware",     # 3
    "Benign"        # 4
]

# ──────── FEATURE EXTRACTION ──────── #

feature_names = list(scaler.feature_names_in_)

def extract_features(url: str, feature_names) -> pd.DataFrame:
    """Extract features from URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ''

    metrics = {}
    try:
        ipaddress.ip_address(hostname)
        metrics['UsingIP'] = 1
    except ValueError:
        metrics['UsingIP'] = 0

    metrics['LongURL'] = len(url)
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co']
    metrics['ShortURL'] = int(any(s in hostname for s in shorteners))
    metrics['Symbol@'] = int('@' in url)
    metrics['Redirecting//'] = int('//' in (parsed.path or ''))
    metrics['PrefixSuffix-'] = hostname.count('-')
    metrics['SubDomains'] = hostname.count('.')
    metrics['HTTPS'] = int(parsed.scheme == 'https')

    # Default all to 0
    df = pd.DataFrame([{name: 0 for name in feature_names}])
    for key, val in metrics.items():
        if key in feature_names:
            df.at[0, key] = val
    return df


def extract_static_features(apk_path):
    static_counts = Counter()
    try:
        a, d, _ = AnalyzeAPK(apk_path)

        # Count permissions
        for perm in a.get_permissions():
            if perm in STATIC_FEATURES:
                static_counts[perm] += 1

        # Count occurrences of each static feature string in all DEX files
        for dex in a.get_all_dex():
            raw = dex if isinstance(dex, (bytes, bytearray)) else dex.get_buff()
            for feat in STATIC_FEATURES:
                occurrences = raw.count(feat.encode())
                if occurrences > 0:
                    static_counts[feat] += occurrences

        return static_counts
    except Exception as e:
        print(f"[!] Static analysis failed: {e}")
        return static_counts

def extract_pseudo_dynamic_features(apk_path):
    pseudo_counts = Counter()
    try:
        with zipfile.ZipFile(apk_path, 'r') as zipf:
            with tempfile.TemporaryDirectory() as temp_dir:
                zipf.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.so', '.dex')):
                            file_path = os.path.join(root, file)
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                for feat in PSEUDO_FEATURES:
                                    occurrences = content.count(feat.encode())
                                    if occurrences > 0:
                                        pseudo_counts[feat] += occurrences

        return pseudo_counts
    except Exception as e:
        print(f"[!] Pseudo-dynamic analysis failed: {e}")
        return pseudo_counts

def generate_vector(static_counts, pseudo_counts):
    # Vector where each element is count from static + pseudo counts
    return np.array(
        [static_counts.get(feat, 0) + pseudo_counts.get(feat, 0)
         for feat in ALL_FEATURES],
        dtype='float32'
    )

# ──────── PREDICTION LOGIC ──────── #
def predict_all_models(feature_vector):
    t = feature_vector.reshape(1, -1)
    X_ml = apk_scaler.transform(t)
    X_dl_repr= X_ml.reshape(1, 235, 1)

    # Use already-loaded models
    probs_rf = rf.predict_proba(X_ml)
    probs_et = et.predict_proba(X_ml)
    probs_gb = gb.predict_proba(X_ml)
    probs_ab = ab.predict_proba(X_ml)
    probs_dl = dl_model.predict(X_dl_repr)

    all_probs = np.array([
        probs_rf[0],
        probs_et[0],
        probs_gb[0],
        probs_ab[0],
        probs_dl[0]
    ])
    avg_probs = np.mean(all_probs, axis=0)
    final_prediction = int(np.argmax(avg_probs))

    return final_prediction, avg_probs

# ──────── FLASK ROUTE ──────── #
@app.route('/upload_apk', methods=['POST'])
def upload_apk():
    if 'file' not in request.files:
        return jsonify({'error': 'No APK provided under key "file"'}), 400

    apk_file = request.files['file']
    if not apk_file.filename.lower().endswith('.apk'):
        return jsonify({'error': 'Uploaded file is not an APK'}), 400

    # Save the APK to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
        apk_path = tmp.name
        apk_file.save(apk_path)

    try:
        static_feats = extract_static_features(apk_path)
        pseudo_feats = extract_pseudo_dynamic_features(apk_path)
        vector       = generate_vector(static_feats, pseudo_feats)
        print(vector)
        label_idx, avg_probs = predict_all_models(vector)
        prob_list = avg_probs.tolist()   # Convert to plain Python list
        print(LABEL_MAP[label_idx])
        result = {
            'predicted_class': LABEL_MAP[label_idx],
            'class_index': label_idx,
            'class_probabilities': dict(zip(LABEL_MAP, prob_list)),
            'features': dict(zip(ALL_FEATURES, vector.tolist()))
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        os.remove(apk_path)


@app.route('/check_url', methods=['POST'])
def check_url():
    url = request.form.get('url')
    print(url)
    if not url:
        return jsonify({'error': 'Missing "url" in request'}), 400

    feats = extract_features(url, feature_names)
    scaled = scaler.transform(feats[feature_names])
    pred = model.predict(scaled)[0]
    label = 'Phishing' if pred == 1 else 'Legitimate'

    return jsonify({
        'url': url,
        'prediction': label,
    })





# ──────── RUN SERVER ──────── #
if __name__ == '__main__':
    # In production, set debug=False
    app.run(host='0.0.0.0', port=5000, debug=True)
