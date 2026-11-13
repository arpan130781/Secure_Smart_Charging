cmake_minimum_required(VERSION 3.14)
project(SecChargeCore C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(seccharge_static STATIC
    src/cusum.c
    src/kalman_chisq.c
    src/water_filling_scheduler.c
    src/secoc.c
)
target_include_directories(seccharge_static PUBLIC include)

add_library(seccharge SHARED
    src/cusum.c
    src/kalman_chisq.c
    src/water_filling_scheduler.c
    src/secoc.c
)
target_include_directories(seccharge PUBLIC include)

enable_testing()
add_executable(test_runner tests/test_runner.c)
target_link_libraries(test_runner PRIVATE seccharge_static)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_runner PRIVATE m)
endif()
add_test(NAME core_unit_tests COMMAND test_runner)
