find_program(PYLINT_EXECUTABLE
             NAMES pylint
             DOC "Path to pylint executable")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(pylint
                                  "Failed to find pylint executable"
                                  PYLINT_EXECUTABLE)
file(GLOB SRCS python/*.py )

add_custom_target(
        pylint
        COMMAND pylint
        --fail-under=8.95
        ${SRCS}
)
