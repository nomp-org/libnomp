function(find_program_path program )
    find_program(EXECUTABLE_PATH
                 NAMES ${program}
                 DOC "Path to ${program} executable")

    include(FindPackageHandleStandardArgs)

    find_package_handle_standard_args(${program}
                                      "Failed to find ${program} executable"
                                      EXECUTABLE_PATH)
endfunction()
