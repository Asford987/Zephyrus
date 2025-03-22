#-------------------------------------------------------------------------------
# Tests (Optional)
#-------------------------------------------------------------------------------

if(VORTEX_INCLUDE_TESTS)
    include(third_party/cmake-scripts/code-coverage.cmake)

    if(TARGET gtest)
        message(FATAL_ERROR "Unit tests can only be built if MLIR is installed.")
    endif()

    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
endif()