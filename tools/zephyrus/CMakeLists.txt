add_executable(zephyrus main.cpp)
target_link_libraries(zephyrus PRIVATE zephyrus_core)

install(TARGETS zephyrus RUNTIME DESTINATION bin)
