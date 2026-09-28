/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

// Linux support used by the imported NET transport, adapted from NCCL's
// src/os/linux.cc. Core initialization, topology and shared-memory helpers
// are intentionally not included in this standalone network plugin.
#include "os.h"
#include "checks.h"
#include "socket.h"
#include "utils.h"

#include <atomic>
#include <dlfcn.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/tcp.h>
#include <unistd.h>

static thread_local char ncclDlErrorBuf[256] = {};

static void saveDlError() {
  const char* err = dlerror();
  snprintf(ncclDlErrorBuf, sizeof(ncclDlErrorBuf), "%s", err ? err : "");
}

ncclOsLibraryHandle ncclOsDlopen(const char* filename) {
  ncclOsLibraryHandle handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) saveDlError();
  return handle;
}

ncclOsLibraryHandle ncclOsDlopen(const char* path, int mode) {
  ncclOsLibraryHandle handle = dlopen(path, mode == NCCL_OS_DL_NOW ? RTLD_NOW : RTLD_LAZY);
  if (handle == nullptr) saveDlError();
  return handle;
}

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  void* ptr = dlsym(handle, symbol);
  if (ptr == nullptr) saveDlError();
  return ptr;
}

void ncclOsDlclose(ncclOsLibraryHandle handle) {
  if (handle) dlclose(handle);
}

const char* ncclOsDlerror() {
  return ncclDlErrorBuf;
}

uint64_t ncclOsGetPid() {
  return static_cast<uint64_t>(getpid());
}

size_t ncclOsGetPageSize() {
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

void ncclOsSetEnv(const char* name, const char* value) {
  setenv(name, value, 0);
}

char* ncclOsStrSep(char** stringp, const char* delim) {
  return strsep(stringp, delim);
}

bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor fd) {
  return fd >= 0;
}

bool ncclOsSocketIsValid(struct ncclSocket* sock) {
  return ncclOsSocketDescriptorIsValid(sock->socketDescriptor);
}

extern int64_t ncclParamPollTimeOut();
extern int64_t ncclParamRetryTimeOut();
extern int64_t ncclParamRetryCnt();
extern int64_t ncclParamSocketMaxRecvBuff();
extern int64_t ncclParamSocketMaxSendBuff();

void ncclOsPollSocket(int fd, int op) {
  struct pollfd pfd = {};
  pfd.fd = fd;
  pfd.events = op == NCCL_SOCKET_RECV ? POLLIN : POLLOUT;
  (void)poll(&pfd, 1, ncclParamPollTimeOut());
}

ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock) {
  socklen_t len = sizeof(union ncclSocketAddress);
  sock->socketDescriptor = accept(sock->acceptSocketDescriptor, &sock->addr.sa, &len);
  if (ncclOsSocketIsValid(sock)) {
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN || errno == ENONET ||
             errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH || errno == EINTR) {
    if (++sock->errorRetries == ncclParamRetryCnt()) {
      WARN("ncclOsSocketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries,
           strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET | NCCL_INIT, "Call to accept returned %s, retrying", strerror(errno));
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    WARN("ncclOsSocketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock) {
  if (!ncclOsSocketIsValid(sock)) return ncclInvalidArgument;
  if (sock->asyncFlag || sock->abortFlag) {
    int flags;
    SYSCHECK(flags = fcntl(sock->socketDescriptor, F_GETFL), "fcntl");
    SYSCHECK(fcntl(sock->socketDescriptor, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }
  const int one = 1;
  SYSCHECK(setsockopt(sock->socketDescriptor, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)), "setsockopt TCP_NODELAY");
  int rcvBuf = ncclParamSocketMaxRecvBuff();
  int sndBuf = ncclParamSocketMaxSendBuff();
  if (sndBuf > 0) {
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_SNDBUF, &sndBuf, sizeof(sndBuf)), "setsockopt SO_SNDBUF");
  }
  if (rcvBuf > 0) {
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_RCVBUF, &rcvBuf, sizeof(rcvBuf)), "setsockopt SO_RCVBUF");
  }
  return ncclSuccess;
}

void ncclOsSocketResetAccept(struct ncclSocket* sock) {
  (void)close(sock->socketDescriptor);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateBadHandshake;
  sock->finalizeCounter = 0;
}

ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  int fd = NCCL_INVALID_SOCKET;
  SYSCHECKGOTO(fd = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  if (ncclOsSocketIsValid(sock)) {
    SYSCHECKGOTO(dup2(fd, sock->socketDescriptor), "dup2", ret, cleanup);
    SYSCHECKGOTO(close(fd), "close", ret, cleanup);
  } else {
    sock->socketDescriptor = fd;
  }
  return ncclOsSocketSetFlags(sock);
cleanup:
  if (fd != NCCL_INVALID_SOCKET) (void)close(fd);
  return ret;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char* funcName) {
  char line[SOCKET_NAME_MAXLEN + 1];
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == EINTR || errCode == EWOULDBLOCK || errCode == EAGAIN || errCode == ETIMEDOUT ||
             errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts", funcName,
             ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET | NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec", funcName,
           ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    }
    NCCLCHECK(ncclOsSocketResetFd(sock));
    sock->state = ncclSocketStateConnecting;
  } else {
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed: %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock) {
  int ret = connect(sock->socketDescriptor, &sock->addr.sa, sock->salen);
  return socketConnectCheck(sock, ret == -1 ? errno : 0, __func__);
}

ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd = {};
  pfd.fd = sock->socketDescriptor;
  pfd.events = POLLOUT;
  int ret = poll(&pfd, 1, 1);
  if (ret == 0 || (ret < 0 && errno == EINTR)) return ncclSuccess;
  if (ret < 0) {
    WARN("ncclOsSocketPollConnect failed: %s", strerror(errno));
    return ncclSystemError;
  }
  socklen_t len = sizeof(ret);
  SYSCHECK(getsockopt(sock->socketDescriptor, SOL_SOCKET, SO_ERROR, &ret, &len), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block,
                                    int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = static_cast<char*>(ptr);
  char line[SOCKET_NAME_MAXLEN + 1];
  if (sock->asyncFlag || sock->abortFlag) block = 0;
  do {
    if (op == NCCL_SOCKET_RECV) {
      bytes = recv(sock->socketDescriptor, data + *offset, size - *offset, block ? 0 : MSG_DONTWAIT);
    } else if (op == NCCL_SOCKET_SEND) {
      bytes = send(sock->socketDescriptor, data + *offset, size - *offset,
                   block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    }
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == -1) {
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;
        return ncclSuccess;
      }
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("ncclOsSocketProgressOpt: %s %s failed: %s", op == NCCL_SOCKET_RECV ? "recv from" : "send to",
             ncclSocketToString(&sock->addr, line), strerror(errno));
        return ncclRemoteError;
      }
      bytes = 0;
    }
    *offset += bytes;
    if (sock->abortFlag &&
        std::atomic_load_explicit((std::atomic<uint32_t>*)sock->abortFlag, std::memory_order_acquire)) {
      INFO(NCCL_NET, "ncclOsSocketProgressOpt: abort called");
      return ncclInternalError;
    }
  } while (sock->asyncFlag == 0 && bytes > 0 && *offset < size);
  return ncclSuccess;
}

ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress* addrs, int sockFamily,
                                  int maxIfNameSize, int maxIfs, int* found) {
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);
  *found = 0;
  struct ifaddrs* interfaces;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (struct ifaddrs* iface = interfaces; iface && *found < maxIfs; iface = iface->ifa_next) {
    if (iface->ifa_addr == nullptr) continue;
    int family = iface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;
    if (!(iface->ifa_flags & IFF_RUNNING)) continue;
    if (sockFamily != -1 && family != sockFamily) continue;
    if (family == AF_INET6 && IN6_IS_ADDR_LOOPBACK(&((struct sockaddr_in6*)iface->ifa_addr)->sin6_addr)) continue;
    if (!(matchIfList(iface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) continue;
    bool duplicate = false;
    for (int i = 0; i < *found; i++) {
      if (strcmp(iface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    snprintf(names + *found * maxIfNameSize, maxIfNameSize, "%s", iface->ifa_name);
    size_t len = family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memset(addrs + *found, 0, sizeof(*addrs));
    memcpy(addrs + *found, iface->ifa_addr, len);
    (*found)++;
  }
  freeifaddrs(interfaces);
  return ncclSuccess;
}

static bool matchSubnet(const struct ifaddrs* local, const union ncclSocketAddress* remote) {
  int family = local->ifa_addr->sa_family;
  if (family != remote->sa.sa_family || local->ifa_netmask == nullptr) return false;
  if (family == AF_INET) {
    auto* addr = (struct sockaddr_in*)local->ifa_addr;
    auto* mask = (struct sockaddr_in*)local->ifa_netmask;
    return (addr->sin_addr.s_addr & mask->sin_addr.s_addr) ==
           (remote->sin.sin_addr.s_addr & mask->sin_addr.s_addr);
  }
  if (family == AF_INET6) {
    auto* addr = (struct sockaddr_in6*)local->ifa_addr;
    auto* mask = (struct sockaddr_in6*)local->ifa_netmask;
    for (int i = 0; i < 16; i++) {
      if ((addr->sin6_addr.s6_addr[i] & mask->sin6_addr.s6_addr[i]) !=
          (remote->sin6.sin6_addr.s6_addr[i] & mask->sin6_addr.s6_addr[i])) return false;
    }
    return addr->sin6_scope_id == remote->sin6.sin6_scope_id;
  }
  return false;
}

ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
  *found = 0;
  struct ifaddrs* interfaces;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (struct ifaddrs* iface = interfaces; iface && !*found; iface = iface->ifa_next) {
    if (iface->ifa_addr == nullptr || !matchSubnet(iface, remoteAddr)) continue;
    size_t len = iface->ifa_addr->sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memset(localAddr, 0, sizeof(*localAddr));
    memcpy(localAddr, iface->ifa_addr, len);
    snprintf(ifName, ifNameMaxSize, "%s", iface->ifa_name);
    *found = 1;
  }
  freeifaddrs(interfaces);
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock == nullptr) return ncclSuccess;
  if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && ncclOsSocketIsValid(sock)) {
    if (wait) {
      char data;
      int closed = 0;
      do {
        int offset = 0;
        if (ncclSocketProgress(NCCL_SOCKET_RECV, sock, &data, sizeof(data), &offset, &closed) != ncclSuccess) break;
      } while (!closed);
    }
    (void)shutdown(sock->socketDescriptor, SHUT_RDWR);
    (void)close(sock->socketDescriptor);
  }
  sock->state = ncclSocketStateClosed;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  return ncclSuccess;
}

ncclResult_t ncclOsNvmlOpen(ncclOsLibraryHandle* handle) {
  *handle = ncclOsDlopen("libnvidia-ml.so.1");
  if (*handle == nullptr) {
    WARN("Failed to open libnvidia-ml.so.1: %s", ncclOsDlerror());
    return ncclSystemError;
  }
  INFO(NCCL_INIT, "Loaded NVML from libnvidia-ml.so.1");
  return ncclSuccess;
}