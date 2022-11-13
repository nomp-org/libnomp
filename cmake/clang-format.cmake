find_program_path(clang-format)

file(GLOB SRCS src/*.c src/*.h)
file(GLOB TESTS tests/*.c tests/*.h)

add_custom_target(
        clangformat
        COMMAND clang-format
        -i
        ${SRCS} ${TESTS}
)
