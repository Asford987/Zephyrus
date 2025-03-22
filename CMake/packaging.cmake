#-------------------------------------------------------------------------------
# Install Directories (Packaging)
#-------------------------------------------------------------------------------

install(DIRECTORY include/ DESTINATION runtime/include)
install(DIRECTORY lib/ DESTINATION runtime/src)
install(FILES ${CMAKE_BINARY_DIR}/runtime/lib/linux/Vortex.so DESTINATION runtime/lib/linux)

#-------------------------------------------------------------------------------
# Debian Package Creation (APT)
#-------------------------------------------------------------------------------

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "vortex-compiler")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_CONTACT "fabricio.asfora.001@gmail.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fabricio Asfora")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6")  # No MLIR/LLVM dependency at runtime
set(CPACK_PACKAGE_DESCRIPTION "Vortex Neural Network Compiler")
set(CPACK_DEBIAN_ARCHITECTURE "i386")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_ARCHITECTURE}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "runtime")

include(CPack)