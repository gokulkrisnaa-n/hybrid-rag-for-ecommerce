#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server for the app.
#
#   sudo deploy/setup_server.sh
#
# Installs Docker, adds a 2 GB swap file, and turns on automatic security
# updates. Safe to run again: each step checks whether it's already done.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo deploy/setup_server.sh" >&2
  exit 1
fi
USER_NAME="${SUDO_USER:-ubuntu}"

echo "→ system updates"
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get upgrade -yq

echo "→ Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
usermod -aG docker "$USER_NAME"          # docker without sudo (after logging in again)
systemctl enable --now docker

echo "→ 2 GB swap (headroom for builds on a 2 GB machine)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -q vm.swappiness=10
grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf

echo "→ automatic security updates"
DEBIAN_FRONTEND=noninteractive apt-get install -yq unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades

echo
echo "Done. Log out and back in (exit, then ssh again) so docker works without sudo."