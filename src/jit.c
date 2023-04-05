#include "nomp-impl.h"

#include <dlfcn.h>
#include <errno.h>
#include <openssl/sha.h>
#include <unistd.h>

static int make_knl_dir(char **dir_, const char *knl_dir, const char *src) {
  size_t len = strnlen(src, MAX_SRC_SIZE);
  if (len == MAX_SRC_SIZE) {
    return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                        "Kernel source size exceeds maximum size %u.",
                        MAX_SRC_SIZE);
  }

  // Check if knl_dir exists, otherwise create it.
  if (access(knl_dir, F_OK) == -1) {
    if (mkdir(knl_dir, S_IRWXU) == -1) {
      return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                          "Unable to create directory: %s. Error: %s.", knl_dir,
                          strerror(errno));
    }
  }

  unsigned char *s = nomp_calloc(unsigned char, len + 1);
  for (unsigned i = 0; i < len; i++)
    s[i] = src[i];

  unsigned char md[SHA256_DIGEST_LENGTH], *ret = SHA256(s, len, md);
  if (ret == NULL) {
    return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                        "Unable to calculate SHA256 hash of string: \"%s\".",
                        s);
  }
  nomp_free(s);

  char *hash = nomp_calloc(char, 2 * SHA256_DIGEST_LENGTH + 1);
  for (unsigned i = 0; i < SHA256_DIGEST_LENGTH; i++)
    snprintf(&hash[2 * i], 3, "%02hhX", md[i]);

  nomp_check(nomp_path_len(&len, knl_dir));
  unsigned lmax = nomp_max(2, len, SHA256_DIGEST_LENGTH);

  // Create the folder if it doesn't exist.
  char *dir = *dir_ = nomp_str_cat(3, lmax, knl_dir, "/", hash);
  if (access(dir, F_OK) == -1) {
    if (mkdir(dir, S_IRWXU) == -1) {
      return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                          "Unable to create directory: %s. Error: %s.", dir,
                          strerror(errno));
    }
  }
  nomp_free(hash);

  return 0;
}

static int write_file(const char *path, const char *src) {
  // See if source exist. Otherwise create it.
  if (access(path, F_OK) == -1 && src != NULL) {
    FILE *fp = fopen(path, "w");
    if (fp != NULL) {
      fprintf(fp, "%s", src);
      fclose(fp);
    } else {
      return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                          "Unable to write file: %s.", path);
    }
  }

  return 0;
}

static int compile_aux(const char *cc, const char *cflags, const char *src,
                       const char *out) {
  size_t len;
  nomp_check(nomp_path_len(&len, cc));
  len += strnlen(cflags, MAX_CFLAGS_SIZE) + strlen(src) + strlen(out) + 32;

  char *cmd = nomp_calloc(char, len);
  snprintf(cmd, len, "%s %s %s -o %s", cc, cflags, src, out);
  int ret = system(cmd), failed = 0;
  if (ret == -1) {
    failed =
        nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                     "Got error \"%s\" when trying to execute command: \"%s\".",
                     cmd, strerror(errno));
  } else if (WIFEXITED(ret)) {
    int status = WEXITSTATUS(ret);
    if (status) {
      failed =
          nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                       "Command: \"%s\" exitted with non-zero status code: %d.",
                       cmd, status);
    }
  } else {
    failed = nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                          "Command: \"%s\" was terminated by a signal.", cmd);
  }
  nomp_free(cmd);

  return failed;
}

struct function {
  void *dlh;
  void (*dlf)(void **);
};

static struct function **funcs = NULL;
static unsigned funcs_n = 0, funcs_max = 0;

int jit_compile(int *id, const char *source, const char *cc, const char *cflags,
                const char *entry, const char *wrkdir, const char *srcf,
                const char *libf) {
  char *dir = NULL;
  nomp_check(make_knl_dir(&dir, wrkdir, source));

  size_t ldir;
  nomp_check(nomp_path_len(&ldir, dir));

  size_t max = nomp_max(3, ldir, strnlen(srcf, 64), strnlen(libf, 64));
  char *src = nomp_str_cat(3, max, dir, "/", srcf);
  char *lib = nomp_str_cat(3, max, dir, "/", libf);
  nomp_free(dir);

  nomp_check(write_file(src, source));
  nomp_check(compile_aux(cc, cflags, src, lib));
  nomp_free(src);

  if (id == NULL)
    return 0;

  if (funcs_n == funcs_max) {
    funcs_max += funcs_max / 2 + 1;
    funcs = nomp_realloc(funcs, struct function *, funcs_max);
  }

  void (*dlf)() = NULL;
  void *dlh = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
  if (dlh && entry)
    dlf = dlsym(dlh, entry);
  nomp_free(lib);

  if (dlh == NULL || dlf == NULL) {
    return nomp_set_log(
        NOMP_JIT_FAILURE, NOMP_ERROR,
        "Failed to open object/symbol \"%s\" due to error: \"%s\".", dlerror());
  }

  struct function *f = funcs[funcs_n] = nomp_calloc(struct function, 1);
  f->dlh = dlh, f->dlf = (void (*)(void **))dlf, *id = funcs_n++;

  return 0;
}

int jit_run(int id, void *p[]) {
  if (id >= 0 && id < funcs_n && funcs[id] && funcs[id]->dlf) {
    funcs[id]->dlf(p);
    return 0;
  }

  return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                      "Failed to run program with handle: %d", id);
}

int jit_free(int *id) {
  int fid = *id;
  if (fid >= 0 && fid < funcs_n) {
    struct function *f = funcs[fid];
    dlclose(f->dlh), f->dlh = NULL, f->dlf = NULL, nomp_free(f);
    funcs[fid] = NULL, *id = -1;
    return 0;
  }

  return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR,
                      "Failed to free program with handle: %d", fid);
}
