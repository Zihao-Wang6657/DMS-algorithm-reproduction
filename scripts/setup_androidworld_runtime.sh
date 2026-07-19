#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="$WORK/downloads"
ANDROID_SDK="$WORK/android_sdk"
ANDROID_AVD="$WORK/android_avd"
JDK_DIR="$WORK/jdks/jdk17"

JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
JDK_ARCHIVE="$DOWNLOAD_DIR/temurin-jdk17-linux-x64.tar.gz"
ANDROID_CMDLINE_URL="https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip"
ANDROID_CMDLINE_ARCHIVE="$DOWNLOAD_DIR/commandlinetools-linux-14742923_latest.zip"
ACCESSIBILITY_FORWARDER_URL="https://storage.googleapis.com/android_env-tasks/2024.05.13-accessibility_forwarder.apk"
ACCESSIBILITY_FORWARDER_APK="$DOWNLOAD_DIR/accessibility_forwarder.apk"

ANDROID_PLATFORM_PACKAGE="platforms;android-33"
ANDROID_IMAGE_PACKAGE="system-images;android-33;google_apis;x86_64"
ANDROID_SDK_PACKAGES=(
  "platform-tools"
  "emulator"
  "$ANDROID_PLATFORM_PACKAGE"
  "$ANDROID_IMAGE_PACKAGE"
)

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required host command: $command_name" >&2
    exit 2
  fi
}

download_if_missing() {
  local url="$1"
  local output="$2"
  local partial_output="${output}.partial"
  if [[ -s "$output" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  rm -f "$partial_output"
  curl -L --fail --retry 3 --retry-all-errors --output "$partial_output" "$url"
  mv "$partial_output" "$output"
}

accept_sdk_licenses() {
  local sdkmanager_status=0
  set +o pipefail
  yes | sdkmanager --sdk_root="$ANDROID_HOME" --licenses >/dev/null || sdkmanager_status=$?
  set -o pipefail
  if [[ "$sdkmanager_status" -ne 0 ]]; then
    echo "failed to accept Android SDK licenses automatically" >&2
    return "$sdkmanager_status"
  fi
}

upsert_ini() {
  local file="$1"
  local key="$2"
  local value="$3"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*$|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

install_jdk() {
  local temp_dir extracted_dir
  if [[ -x "$JDK_DIR/bin/java" ]]; then
    return 0
  fi
  download_if_missing "$JDK_URL" "$JDK_ARCHIVE"
  temp_dir="$(mktemp -d)"
  mkdir -p "$JDK_DIR"
  tar -xzf "$JDK_ARCHIVE" -C "$temp_dir"
  extracted_dir="$(find "$temp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  rm -rf "$JDK_DIR"
  mkdir -p "$JDK_DIR"
  cp -a "$extracted_dir"/. "$JDK_DIR"/
  rm -rf "$temp_dir"
}

install_cmdline_tools() {
  local temp_dir tools_dir
  if [[ -x "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" ]]; then
    return 0
  fi
  download_if_missing "$ANDROID_CMDLINE_URL" "$ANDROID_CMDLINE_ARCHIVE"
  temp_dir="$(mktemp -d)"
  unzip -q "$ANDROID_CMDLINE_ARCHIVE" -d "$temp_dir"
  tools_dir="$(find "$temp_dir" -mindepth 1 -maxdepth 2 -type d -name cmdline-tools | head -n 1)"
  if [[ -z "$tools_dir" ]]; then
    echo "failed to locate extracted Android command-line tools" >&2
    exit 2
  fi
  mkdir -p "$ANDROID_SDK/cmdline-tools"
  rm -rf "$ANDROID_SDK/cmdline-tools/latest"
  mv "$tools_dir" "$ANDROID_SDK/cmdline-tools/latest"
  rm -rf "$temp_dir"
}

install_sdk_packages() {
  export JAVA_HOME="$JDK_DIR"
  export ANDROID_HOME="$ANDROID_SDK"
  export ANDROID_SDK_ROOT="$ANDROID_HOME"
  export ANDROID_AVD_HOME="$ANDROID_AVD"
  export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"

  mkdir -p "$ANDROID_HOME" "$ANDROID_AVD_HOME" "$DOWNLOAD_DIR"
  accept_sdk_licenses
  sdkmanager --sdk_root="$ANDROID_HOME" "${ANDROID_SDK_PACKAGES[@]}"
}

create_avd() {
  local avd_ini="$ANDROID_AVD/AndroidWorldAvd.ini"
  local avd_config="$ANDROID_AVD/AndroidWorldAvd.avd/config.ini"
  if [[ ! -f "$avd_ini" ]]; then
    printf 'no\n' | avdmanager create avd \
      --force \
      --name "AndroidWorldAvd" \
      --device "pixel_6" \
      --package "$ANDROID_IMAGE_PACKAGE"
  fi
  upsert_ini "$avd_config" "disk.dataPartition.size" "6G"
  upsert_ini "$avd_config" "hw.ramSize" "2G"
  upsert_ini "$avd_config" "hw.cpu.ncore" "4"
}

main() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "setup_androidworld_runtime.sh currently supports Linux only." >&2
    exit 2
  fi
  case "$(uname -m)" in
    x86_64|amd64) ;;
    *)
      echo "setup_androidworld_runtime.sh expects an x86_64 host for the Android emulator." >&2
      exit 2
      ;;
  esac

  require_command curl
  require_command unzip
  require_command tar

  install_jdk
  install_cmdline_tools
  install_sdk_packages
  download_if_missing "$ACCESSIBILITY_FORWARDER_URL" "$ACCESSIBILITY_FORWARDER_APK"
  create_avd

  for optional_command in tmux; do
    if ! command -v "$optional_command" >/dev/null 2>&1; then
      echo "warning: missing optional host command: $optional_command" >&2
      echo "warning: start_androidworld_emulator.sh will require it." >&2
    fi
  done

  echo "java_home=$JDK_DIR"
  echo "android_sdk=$ANDROID_SDK"
  echo "android_avd=$ANDROID_AVD"
  echo "accessibility_forwarder_apk=$ACCESSIBILITY_FORWARDER_APK"
}

main "$@"
