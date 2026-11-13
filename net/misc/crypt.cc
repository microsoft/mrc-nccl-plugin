// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Plaintext-only socket hello, adapted from NCCL's src/misc/crypt.cc. TLS is
// configured through the public ncclSetEncryption() API of libnccl, which a
// standalone NET plugin cannot reach, so the OpenSSL backend is not included.

#include "crypt.h"
#include "core.h"
#include "os.h"
#include "socket.h"
#include <stdlib.h>
#include <string.h>

extern int64_t ncclParamPollTimeOut();

struct ncclSocketCrypto {
  bool handshakeDone;
};

static void cryptPollSocket(struct ncclSocket* sock, int op) {
  if (sock->asyncFlag == 0 && ncclParamPollTimeOut()) ncclOsPollSocket(sock->socketDescriptor, op);
}

static ncclResult_t cryptSocketProgressRaw(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  int closed = 0;
  NCCLCHECK(ncclOsSocketProgressOpt(op, sock, ptr, size, offset, /*block=*/0, &closed));
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("cryptSocketProgressRaw: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
    return ncclRemoteError;
  }
  return ncclSuccess;
}

ncclResult_t ncclGetCryptConnectionMode(bool* encrypted) {
  *encrypted = false;
  return ncclSuccess;
}

void ncclCryptRebindSocket(struct ncclSocket*) {}

ncclResult_t ncclCryptStartConnect(struct ncclSocket* sock) {
  memcpy(sock->finalizeBuffer, &sock->magic, sizeof(sock->magic));
  memcpy(sock->finalizeBuffer + sizeof(sock->magic), &sock->type, sizeof(sock->type));
  return ncclSuccess;
}

ncclResult_t ncclCryptConnectHello(struct ncclSocket* sock) {
  if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
    NCCLCHECK(cryptSocketProgressRaw(NCCL_SOCKET_SEND, sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES,
                                     &sock->finalizeCounter));
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      cryptPollSocket(sock, NCCL_SOCKET_SEND);
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCryptAcceptHello(struct ncclSocket* sock, enum ncclCryptHelloVerdict* verdict) {
  uint64_t magic;
  enum ncclSocketType type;
  char line[SOCKET_NAME_MAXLEN + 1];
  *verdict = NCCL_CRYPT_HELLO_VERDICT_AGAIN;

  if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
    int closed = 0;
    // A recv error during the hello resets like a close (stock's synchronous
    // path did the same): one flaky connection must not kill the accept loop.
    ncclResult_t res =
      ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES,
                              &sock->finalizeCounter, /*block=*/0, &closed);
    if (res == ncclRemoteError) {
      ATTN("Handshake receive from %s failed, discarding peer connection", ncclSocketToString(&sock->addr, line));
      *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
      return ncclSuccess;
    }
    NCCLCHECK(res);
    if (closed) {
      ATTN("Peer %s closed before socket hello, discarding", ncclSocketToString(&sock->addr, line));
      *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
      return ncclSuccess;
    }
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      cryptPollSocket(sock, NCCL_SOCKET_RECV);
      return ncclInProgress;
    }
  }

  memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
  if (magic != sock->magic) {
    ATTN("Socket magic mismatch from %s (peer 0x%016llx != expected 0x%016llx), discarding peer connection",
         ncclSocketToString(&sock->addr, line), (unsigned long long)magic, (unsigned long long)sock->magic);
    *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
    return ncclSuccess;
  }
  memcpy(&type, sock->finalizeBuffer + sizeof(magic), sizeof(type));
  if (type != sock->type) {
    ATTN("Wrong socket type from %s (peer %d != expected %d -- peer connected to wrong NCCL socket), discarding peer "
         "connection",
         ncclSocketToString(&sock->addr, line), (int)type, (int)sock->type);
    *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
    return ncclSuccess;
  }
  *verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  return ncclSuccess;
}

ncclResult_t ncclCryptSocketSend(struct ncclSocket*, void*, int, int*, bool*) {
  WARN("ncclCryptSocketSend called without TLS support");
  return ncclInternalError;
}

ncclResult_t ncclCryptSocketRecv(struct ncclSocket*, void*, int, int*, bool*) {
  WARN("ncclCryptSocketRecv called without TLS support");
  return ncclInternalError;
}

void ncclCryptFree(struct ncclSocketCrypto* crypto) {
  free(crypto);
}
