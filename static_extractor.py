import os
import re
import json
import zipfile
import tempfile
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.stats import mode
from androguard.misc import AnalyzeAPK
from collections import Counter

# ---------- CONFIG ---------- #
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
    'windowGainedFocus', 'write', 'writev']

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

PSEUDO_FEATURES =[
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

# ---------- FEATURE EXTRACTION ---------- #
def extract_static_features(apk_path):
    found = set()
    try:
        a, d, _ = AnalyzeAPK(apk_path)
        for perm in a.get_permissions():
            if perm in STATIC_FEATURES:
                found.add(perm)
        for dex in a.get_all_dex():
            for feat in STATIC_FEATURES:
                if re.search(re.escape(feat).encode(), dex):
                    found.add(feat)

        # Save to text file
        out_path = os.path.splitext(os.path.basename(apk_path))[0] + "_static_features.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(sorted(found)))
        print(f"[✓] Static features saved to {out_path}")

    except Exception as e:
        print(f"[!] Static analysis failed: {e}")
    return found


def extract_pseudo_dynamic_features(apk_path):
    found = set()
    try:
        with zipfile.ZipFile(apk_path, 'r') as zipf:
            with tempfile.TemporaryDirectory() as temp_dir:
                zipf.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.so', '.dex')):
                            with open(os.path.join(root, file), 'rb') as f:
                                content = f.read()
                                for feat in PSEUDO_FEATURES:
                                    if feat.encode() in content:
                                        found.add(feat)

        # Save to text file
        out_path = os.path.splitext(os.path.basename(apk_path))[0] + "_pseudo_features.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(sorted(found)))
        print(f"[✓] Pseudo-dynamic features saved to {out_path}")

    except Exception as e:
        print(f"[!] Pseudo-dynamic analysis failed: {e}")
    return found


def generate_vector(static_feats, pseudo_feats):
    return np.array(
        [1 if feat in (static_feats | pseudo_feats) else 0 for feat in ALL_FEATURES],
        dtype='float32'
    )


# ---------- MODEL PREDICTION ---------- #
def predict_all_models(feature_vector):
    scaler = joblib.load("scaler_cnn_bilstm_initial.pkl")
    t = feature_vector.reshape(1, -1)
    X_ml = scaler.transform(t)
    X_dl_scaled = X_ml.reshape(1, 235, 1)

    # Load classical models
    rf = joblib.load("Random_Forest.pkl")
    et = joblib.load("Extra_Trees.pkl")
    gb = joblib.load("Gradient_Boosting_Machine.pkl")
    ab = joblib.load("AdaBoost.pkl")

    # Load DL model
    dl_model = load_model("model_bilstm_cnn_best.keras")

    # Get prediction probabilities from each model
    probs_rf = rf.predict_proba(X_ml)
    probs_et = et.predict_proba(X_ml)
    probs_gb = gb.predict_proba(X_ml)
    probs_ab = ab.predict_proba(X_ml)
    probs_dl = dl_model.predict(X_dl_scaled)

    # Stack and average probabilities
    all_probs = np.array([probs_rf[0], probs_et[0], probs_gb[0], probs_ab[0], probs_dl[0]])
    avg_probs = np.mean(all_probs, axis=0)

    final_prediction = np.argmax(avg_probs)

    print("Individual class probabilities:")
    for i, probs in enumerate(all_probs, start=1):
        print(f"Model {i}: {probs}")

    label_map = [
        "Adware",  # 0
        "Banking",  # 1
        "SMS malware",  # 2
        "Riskware",  # 3
        "Benign"  # 4
    ]

    print("Average probabilities:", avg_probs)
    print("Final Prediction (Soft Voting):", label_map[final_prediction])
    return final_prediction


# ---------- MAIN ---------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python full_predictor.py <apk_file>")
        sys.exit(1)

    apk_path = sys.argv[1]
    print("[*] Extracting features from:", apk_path)

    static_features = extract_static_features(apk_path)
    pseudo_features = extract_pseudo_dynamic_features(apk_path)
    vector = generate_vector(static_features, pseudo_features)

    print(vector)
    print(len(vector))
    predict_all_models(vector)
