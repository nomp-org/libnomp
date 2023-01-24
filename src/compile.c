#include <openssl/sha.h>

#include "nomp-impl.h"

static void get_str_hash(unsigned char hash[SHA256_DIGEST_LENGTH], const char *str) {
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  SHA256_Update(&ctx, str, strlen(str)*sizeof(char));
  SHA256_Final(hash, &ctx);
}

void compile(char *path) {
  char *nomp_dir = getenv("NOMP_INSTALL_DIR");
  char *cmd =
      strcatn(5, BUFSIZ, "icpx -o ", nomp_dir, "/lib/libkernellib.o -c -fPIC ",
              path, " -fsycl -fsycl-unnamed-lambda ");
  int err = system(cmd);
  cmd = strcatn(5, BUFSIZ, "icpx -shared -o ", nomp_dir,
                "/lib/libkernellib.so.0.0.1 ", nomp_dir,
                "/lib/libkernellib.o -fsycl -fsycl-unnamed-lambda ");
  err = system(cmd);
}

char *writefile(const char *source) {
  char filename[] = "mylib.cpp";
  FILE *fptr = fopen(filename, "w");
  if (fptr == NULL) {
    printf("Error! opening file");
    // Program exits if the file pointer returns NULL.
    exit(1);
  }

  fprintf(fptr, "%s", source);
  fclose(fptr);
  return realpath(filename, NULL);
}
