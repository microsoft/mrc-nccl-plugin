// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <mutex>
#include <unordered_set>
#include "os.h"

const char* userHomeDir() {
  return getenv("HOME");
}

void setEnvFile(const char* fileName) {
  FILE* file = fopen(fileName, "r");
  if (file == NULL) return;

  char line[4096];
  char envVar[1024];
  char envValue[1024];
  while (fgets(line, (int)sizeof(line), file) != NULL) {
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
    if (len > 0 && line[len - 1] == '\r') line[--len] = '\0';
    if (line[0] == '#') continue;
    int s = 0;
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023, s));
    envVar[std::min(1023, s)] = '\0';
    s++;
    strncpy(envValue, line + s, 1023);
    envValue[1023] = '\0';
    ncclOsSetEnv(envVar, envValue);
  }
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = std::getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  static std::once_flag once;
  std::call_once(once, initEnvFunc);
}

static void ncclGetCachePolicy(char const* env, int8_t* noCache) {
  // The NCCL core's parameter registry and environment plugin are private.
  // Keep the standalone plugin's cache policy and config-file handling local.
  static std::once_flag once;
  static std::unordered_set<std::string> uncached;
  std::call_once(once, []() {
    const char* value = ncclGetEnv("NCCL_NO_CACHE");
    if (value == nullptr) return;
    std::string list(value);
    size_t start = 0;
    while (start < list.size()) {
      size_t end = list.find(',', start);
      std::string key = list.substr(start, end - start);
      size_t first = key.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) {
        size_t last = key.find_last_not_of(" \t\r\n");
        uncached.insert(key.substr(first, last - first + 1));
      }
      if (end == std::string::npos) break;
      start = end + 1;
    }
  });
  *noCache = strcmp(env, "NCCL_NO_CACHE") != 0 && (uncached.count("ALL") || uncached.count(env));
}

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // noCache is only load/stored within the mutex, no need for atomic
  if (*noCache == /*uninitialized*/ -1) ncclGetCachePolicy(env, noCache);

  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) != uninitialized) {
    return COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed);
  }

  // Read the environment variable
  const char* str = ncclGetEnv(env);
  int64_t value = deftVal;

  if (str && strlen(str) > 0) {
    errno = 0;
    value = strtoll(str, nullptr, 0);
    if (errno) {
      value = deftVal;
      INFO(NCCL_ALL, "Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
    } else {
      INFO(NCCL_ENV, "%s set by environment to %lld.", env, (long long)value);
    }
  }

  if (*noCache == /*cache*/ 0) COMPILER_ATOMIC_STORE(cache, value, std::memory_order_relaxed);
  return value;
}

const char* ncclGetEnv(const char* name) {
  initEnv();
  return std::getenv(name);
}
