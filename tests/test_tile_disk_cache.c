#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>
#include "viz/tile_disk_cache.h"

static void fail(const char *message) {
  fprintf(stderr, "!!! %s\n", message);
  exit(EXIT_FAILURE);
}

static void assert_true(bool condition, const char *message) {
  printf("%s -> %s\n", message, condition ? "ok" : "fail");
  if (!condition) fail(message);
}

static void assert_size_eq(size_t expected, size_t actual, const char *message) {
  printf("%s: %zu == %zu\n", message, expected, actual);
  if (expected != actual) {
    fprintf(stderr, "!!! %s: expected %zu but got %zu\n",
            message, expected, actual);
    exit(EXIT_FAILURE);
  }
}

static void create_temp_dir(char *path_template) {
  int fd = mkstemp(path_template);
  if (fd < 0) {
    perror("mkstemp");
    exit(EXIT_FAILURE);
  }
  close(fd);
  unlink(path_template);
  if (mkdir(path_template, 0700) != 0) {
    perror("mkdir");
    exit(EXIT_FAILURE);
  }
}

static void remove_tree(const char *path) {
  DIR *dir;
  struct dirent *entry;

  if (!path) return;
  dir = opendir(path);
  if (!dir) {
    if (errno == ENOENT) return;
    perror(path);
    exit(EXIT_FAILURE);
  }

  while ((entry = readdir(dir)) != NULL) {
    char child[PATH_MAX];
    struct stat st;

    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) >=
        (int)sizeof(child)) {
      closedir(dir);
      fail("temp path too long");
    }
    if (lstat(child, &st) != 0) {
      closedir(dir);
      perror(child);
      exit(EXIT_FAILURE);
    }
    if (S_ISDIR(st.st_mode)) {
      remove_tree(child);
    } else if (unlink(child) != 0) {
      closedir(dir);
      perror(child);
      exit(EXIT_FAILURE);
    }
  }

  closedir(dir);
  if (rmdir(path) != 0) {
    perror(path);
    exit(EXIT_FAILURE);
  }
}

static bool find_first_file_recursive(const char *dir_path, char *out,
                                      size_t out_size) {
  DIR *dir;
  struct dirent *entry;

  dir = opendir(dir_path);
  if (!dir) return false;

  while ((entry = readdir(dir)) != NULL) {
    char child[PATH_MAX];
    struct stat st;
    bool found = false;

    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (snprintf(child, sizeof(child), "%s/%s", dir_path, entry->d_name) >=
        (int)sizeof(child)) {
      closedir(dir);
      fail("temp path too long");
    }
    if (stat(child, &st) != 0) {
      continue;
    }
    if (S_ISDIR(st.st_mode)) {
      found = find_first_file_recursive(child, out, out_size);
    } else if (S_ISREG(st.st_mode)) {
      found = snprintf(out, out_size, "%s", child) < (int)out_size;
    }
    if (found) {
      closedir(dir);
      return true;
    }
  }

  closedir(dir);
  return false;
}

static void test_write_and_read_round_trip(void) {
  char dir_template[] = "/tmp/psm_tile_disk_cache_roundtrip_XXXXXX";
  char root[PATH_MAX];
  TileDiskCache cache;
  uint8_t payload[] = {1, 3, 3, 7};
  uint8_t *loaded = NULL;
  size_t loaded_size = 0;

  create_temp_dir(dir_template);
  assert_true(snprintf(root, sizeof(root), "%s/cache", dir_template) <
                  (int)sizeof(root),
              "build roundtrip cache root");

  TileDiskCache_init(&cache, "CartoDB.Positron",
                     "https://tiles.example.com/{z}/{x}/{y}.png");
  assert_true(TileDiskCache_use_root(&cache, root), "enable disk cache at root");
  TileDiskCache_configure(&cache, true, 1024);

  assert_true(TileDiskCache_write(&cache, 1, 2, 3, payload, sizeof(payload)),
              "write tile payload");
  assert_size_eq(1, (size_t)cache.writes, "write count");
  assert_size_eq(sizeof(payload), cache.bytes, "cache bytes after write");

  assert_true(TileDiskCache_read(&cache, 1, 2, 3, &loaded, &loaded_size),
              "read tile payload");
  assert_size_eq(1, (size_t)cache.hits, "cache hit count");
  assert_size_eq(sizeof(payload), loaded_size, "loaded byte count");
  assert_true(memcmp(payload, loaded, sizeof(payload)) == 0,
              "round-trip payload matches");

  free(loaded);
  remove_tree(dir_template);
}

static void test_prune_oldest_file_to_fit_budget(void) {
  char dir_template[] = "/tmp/psm_tile_disk_cache_prune_XXXXXX";
  char root[PATH_MAX];
  char first_path[PATH_MAX];
  TileDiskCache cache;
  uint8_t old_payload[] = {1, 2, 3, 4, 5};
  uint8_t new_payload[] = {6, 7, 8, 9, 10};
  struct utimbuf old_time = {0};
  uint8_t *loaded = NULL;
  size_t loaded_size = 0;

  create_temp_dir(dir_template);
  assert_true(snprintf(root, sizeof(root), "%s/cache", dir_template) <
                  (int)sizeof(root),
              "build prune cache root");

  TileDiskCache_init(&cache, "CartoDB.DarkMatter",
                     "https://tiles.example.com/{z}/{x}/{y}.png");
  assert_true(TileDiskCache_use_root(&cache, root), "enable prune cache root");
  TileDiskCache_configure(&cache, true, 8);

  assert_true(TileDiskCache_write(&cache, 3, 4, 5,
                                  old_payload, sizeof(old_payload)),
              "write old tile");
  assert_true(find_first_file_recursive(root, first_path, sizeof(first_path)),
              "locate old tile path");
  old_time.actime = time(NULL) - 3600;
  old_time.modtime = old_time.actime;
  assert_true(utime(first_path, &old_time) == 0, "age old tile");

  assert_true(TileDiskCache_write(&cache, 9, 10, 5,
                                  new_payload, sizeof(new_payload)),
              "write new tile and prune");
  assert_size_eq(1, (size_t)cache.prunes, "prune count");
  assert_size_eq(sizeof(new_payload), cache.bytes, "cache bytes after prune");
  assert_true(!TileDiskCache_read(&cache, 3, 4, 5, &loaded, &loaded_size),
              "old tile pruned");
  assert_true(TileDiskCache_read(&cache, 9, 10, 5, &loaded, &loaded_size),
              "new tile retained");
  assert_true(memcmp(new_payload, loaded, sizeof(new_payload)) == 0,
              "new payload matches");

  free(loaded);
  remove_tree(dir_template);
}

static void test_disabled_cache_rejects_io(void) {
  char dir_template[] = "/tmp/psm_tile_disk_cache_disabled_XXXXXX";
  char root[PATH_MAX];
  TileDiskCache cache;
  uint8_t payload[] = {42};
  uint8_t *loaded = NULL;
  size_t loaded_size = 0;

  create_temp_dir(dir_template);
  assert_true(snprintf(root, sizeof(root), "%s/cache", dir_template) <
                  (int)sizeof(root),
              "build disabled cache root");

  TileDiskCache_init(&cache, "CartoDB.PositronNoLabels",
                     "https://tiles.example.com/{z}/{x}/{y}.png");
  assert_true(TileDiskCache_use_root(&cache, root), "enable disabled cache root");
  TileDiskCache_configure(&cache, false, 1024);

  assert_true(!TileDiskCache_write(&cache, 0, 0, 0, payload, sizeof(payload)),
              "disabled cache write rejected");
  assert_true(!TileDiskCache_read(&cache, 0, 0, 0, &loaded, &loaded_size),
              "disabled cache read rejected");
  assert_size_eq(0, cache.bytes, "disabled cache bytes");

  remove_tree(dir_template);
}

int main(void) {
  test_write_and_read_round_trip();
  test_prune_oldest_file_to_fit_budget();
  test_disabled_cache_rejects_io();
  return 0;
}
