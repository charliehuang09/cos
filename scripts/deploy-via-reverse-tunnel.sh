#!/usr/bin/env bash
# Deploy the COS repository from build-server when build-server cannot reach
# dev-orin directly.
#
# Run this script on a workstation that can reach both hosts:
#
#   ./deploy-via-reverse-tunnel.sh
#
# The script streams its deployment half to build-server, while this SSH
# session exposes dev-orin:22 on build-server's loopback interface.

set -Eeuo pipefail

BUILD_SERVER="${BUILD_SERVER:-yasen@build-server}"
ORIN_HOST="${ORIN_HOST:-dev-orin}"
REMOTE_REPO="${REMOTE_REPO:-/home/yasen/cos}"
TUNNEL_PORT="${TUNNEL_PORT:-22222}"

usage() {
  cat <<'EOF'
Usage: deploy-via-reverse-tunnel.sh

Deploy the configured COS build on build-server to dev-orin through an SSH
reverse tunnel. Run this on a host that can reach both systems.

Configuration is supplied with environment variables:
  BUILD_SERVER  SSH destination for the build server (default: yasen@build-server)
  ORIN_HOST     Orin host reachable from this workstation (default: dev-orin)
  REMOTE_REPO   Repository path on build-server (default: /home/yasen/cos)
  TUNNEL_PORT   Loopback port on build-server (default: 22222)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${COS_DEPLOY_TUNNEL_REMOTE:-0}" != 1 ]]; then
  # -A lets the build-server-side SSH client use this workstation's agent if
  # the Orin requires public-key authentication. The reverse forward itself
  # is created by this workstation, so build-server need not resolve ORIN_HOST.
  exec ssh \
    -A \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R "127.0.0.1:${TUNNEL_PORT}:${ORIN_HOST}:22" \
    "${BUILD_SERVER}" \
    env \
      COS_DEPLOY_TUNNEL_REMOTE=1 \
      COS_DEPLOY_TUNNEL_PORT="${TUNNEL_PORT}" \
      bash -s -- "${REMOTE_REPO}" "${TUNNEL_PORT}" < "$0"
fi

# Everything below runs on build-server inside the SSH session above.
REMOTE_REPO="${1:?missing remote repository path}"
TUNNEL_PORT="${2:?missing tunnel port}"

if [[ ! -d "${REMOTE_REPO}/build" ]]; then
  echo "Build directory not found: ${REMOTE_REPO}/build" >&2
  exit 1
fi

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/cos-deploy-tunnel.XXXXXX")"
trap 'rm -rf -- "${temporary_directory}"' EXIT

# CMake's generated dev-orin target invokes ssh and rsync with the literal
# host name dev-orin. Put wrappers first in PATH so the existing target can be
# reused without changing the generated build files or the user's SSH config.
cat > "${temporary_directory}/ssh" <<'EOF'
#!/usr/bin/env bash
exec /usr/bin/ssh -F "${COS_DEPLOY_SSH_CONFIG}" "$@"
EOF
chmod 700 "${temporary_directory}/ssh"

cat > "${temporary_directory}/ssh-config" <<EOF
Host dev-orin
    HostName 127.0.0.1
    Port ${TUNNEL_PORT}
    HostKeyAlias dev-orin
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ${temporary_directory}/known_hosts
EOF

export COS_DEPLOY_SSH_CONFIG="${temporary_directory}/ssh-config"
PATH="${temporary_directory}:${PATH}" \
  cmake --build "${REMOTE_REPO}/build" --target dev-orin
