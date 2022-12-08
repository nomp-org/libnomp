find_program(PYLINT_EXECUTABLE
             NAMES pylint
             DOC "Path to pylint executable")

find_program(FLAKE8_EXECUTABLE
             NAMES flake8
             DOC "Path to flake8 executable")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(pylint
                                  "Failed to find pylint executable"
                                  PYLINT_EXECUTABLE)
find_package_handle_standard_args(flake8
                                  "Failed to find flake8 executable"
                                  FLAKE8_EXECUTABLE)
file(GLOB SRCS python/*.py tests/*.py)

add_custom_target(
        pylint
        COMMAND pylint
        --fail-under=8.95
        ${SRCS}
)

add_custom_target(
        flake8
        COMMAND flake8
        ${SRCS}
)
