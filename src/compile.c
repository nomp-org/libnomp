#include "nomp-impl.h"

#include <dlfcn.h>
#include <errno.h>
#include <openssl/sha.h>
#include <unistd.h>

static int make_knl_dir(char **dir_, const char *cache_dir, const char *src) {
  unsigned len = strnlen(src, MAX_SRC_SIZE);
  if (len == MAX_SRC_SIZE) {
    return set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                   "Kernel source size exceeds maximum allowed size: %u.",
                   MAX_SRC_SIZE);
  }

  unsigned char *s = tcalloc(unsigned char, len + 1),
                hash[SHA256_DIGEST_LENGTH];
  for (unsigned i = 0; i < len; i++)
    s[i] = src[i];

  unsigned char *ret = SHA256(s, len, hash);
  tfree(s);
  if (ret == NULL) {
    return set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                   "Unable to calculate SHA256 hash of string: \"%s\".", s);
  }

  unsigned max = MAX(pathlen(cache_dir), SHA256_DIGEST_LENGTH);
  char *dir = *dir_ = strcatn(3, max, cache_dir, "/", hash);
  // Create the folder if it doesn't exist.
  if (access(dir, F_OK)) {
    if (mkdir(dir, S_IRUSR | S_IWUSR) == -1) {
      tfree(dir);
      return set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                     "Unable to create directory: %s.", dir);
    }
  }

  return 0;
}

static int write_file(const char *path, const char *src) {
  // See if source.cpp exist. Otherwise create it.
  if (access(path, F_OK)) {
    FILE *fp = fopen(path, "w");
    if (fp != NULL) {
      fprintf(fp, "%s", src);
      fclose(fp);
    } else {
      return set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                     "Unable to write file: %s.", path);
    }
  }

  return 0;
}

static int compile_aux(const char *cc, const char *cflags, const char *src,
                       const char *out) {
  unsigned len = pathlen(cc) + strnlen(cflags, MAX_CFLAGS_SIZE) + pathlen(src) +
                 pathlen(out) + 32;
  char *cmd = tcalloc(char, len);
  snprintf(cmd, len, "%s %s %s -o %s", cc, cflags, src, out);
  int ret = system(cmd), failed = 0;
  if (ret == -1) {
    failed = set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                     "Got error \"%s\" when trying to execute command: \"%s\".",
                     cmd, strerror(errno));
  } else if (WIFEXITED(ret)) {
    int status = WEXITSTATUS(ret);
    if (status) {
      failed = set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                       "Command: \"%s\" exitted with non-zero status code: %d.",
                       cmd, status);
    }
  } else {
    failed = set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                     "Command: \"%s\" was terminated by a signal.", cmd);
  }
  tfree(cmd);

  return failed;
}

struct function {
  void *dlh;
  void (*dlf)(int N, ...);
};

static struct function **funcs = NULL;
static unsigned funcs_n = 0, funcs_max = 0;

int compile(int *id, const char *source, const char *cc, const char *cflags,
            const char *entry, const char *wrkdir) {
  char *dir = NULL;
  return_on_err(make_knl_dir(&dir, wrkdir, source));

  const char *cpp = "source.cpp", *lib = "mylib.so";
  unsigned max = MAX(pathlen(dir), MAX(strnlen(cpp, 64), strnlen(lib, 64)));
  char *src = strcatn(3, max, dir, "/", cpp);
  char *out = strcatn(3, max, dir, "/", lib);
  tfree(dir);

  return_on_err(write_file(src, source));
  return_on_err(compile_aux(cc, cflags, src, out));
  tfree(src), tfree(out);

  if (funcs_n == funcs_max) {
    funcs_max += funcs_max / 2 + 1;
    funcs = trealloc(funcs, struct function *, funcs_max);
  }

  void (*dlf)() = NULL;
  void *dlh = dlopen(out, RTLD_LAZY | RTLD_LOCAL);
  if (dlh)
    dlf = dlsym(dlh, entry);

  if (dlh == NULL || dlf == NULL) {
    return set_log(NOMP_COMPILE_FAILURE, NOMP_ERROR,
                   "Failed to open object/symbol \"%s\". Error: \"%s\".\n",
                   dlerror());
  }

  struct function *f = funcs[funcs_n] = tcalloc(struct function, 1);
  f->dlh = dlh, f->dlf = (void (*)(int N, ...))dlf, *id = funcs_n++;

  return 0;
}
