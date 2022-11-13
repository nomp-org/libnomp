find_program_path(pylint)
file(GLOB SRCS python/*.py )

add_custom_target(
        pylint
        COMMAND pylint
        --fail-under=8.95
        ${SRCS}
)
