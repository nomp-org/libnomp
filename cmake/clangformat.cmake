find_program(CLANG-FORMAT_EXECUTABLE
             NAMES clang-format
             DOC "Path to clang-format executable")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(clang-format
                                  "Failed to find clang-format executable"
                                  CLANG-FORMAT_EXECUTABLE)
file(GLOB SRCS src/*.c src/*.h)
file(GLOB TESTS tests/*.c tests/*.h)

add_custom_target(
        clangformat
        COMMAND clang-format
        -i
        ${SRCS} ${TESTS}
)
