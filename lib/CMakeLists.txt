# At the top of lib/CMakeLists.txt
file(GLOB ZEUS_HANDLER_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/Passes/handlers/*.cpp")

add_library(zephyrus_passes
    Passes/Passes.cpp
    Passes/HDF5ToTosaPass.cpp
    Passes/TosaToLLVMPass.cpp
    ${ZEUS_HANDLER_SRCS}         # ← expanded list
)

target_include_directories(zephyrus_passes
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include       # public headers
)
target_link_libraries(zephyrus_passes
    PUBLIC                                # cascades to dependents
    MLIRIR
    MLIRPass
    MLIRTosa
    MLIRTransforms
)

# ---- Core compiler lib (compileModel, etc.) ----
add_library(zephyrus_core Compiler.cpp)
    target_include_directories(zephyrus_core
        PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(zephyrus_core
        PUBLIC
        zephyrus_passes
        nlohmann_json::nlohmann_json
        HDF5::HDF5
        MLIRIR MLIRSupport LLVMCore LLVMTarget
)

# ---- Install both libs ----
install(TARGETS zephyrus_passes zephyrus_core
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib)
