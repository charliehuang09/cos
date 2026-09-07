include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

set(ALL_BUILD_TARGETS)
foreach(PROJECT_TARGET IN LISTS ALL_PROJECT_TARGETS)
    get_target_property(PROJECT_TARGET_TYPE ${PROJECT_TARGET} TYPE)
    if(NOT PROJECT_TARGET_TYPE STREQUAL "UTILITY"
       AND NOT PROJECT_TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        list(APPEND ALL_BUILD_TARGETS ${PROJECT_TARGET})
    endif()
endforeach()

add_custom_target(dev-orin
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying build/bin and build/lib to root@dev-orin:/root..."
    
    COMMAND ssh root@dev-orin "mkdir -p /root"

    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/tests/ root@dev-orin:/root/tests/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/main/ root@dev-orin:/root/main/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/examples/ root@dev-orin:/root/examples/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/tools/ root@dev-orin:/root/tools/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/lib/ root@dev-orin:/root/lib/
    COMMAND rsync -avz --delete ${CMAKE_SOURCE_DIR}/constants/ root@dev-orin:/root/constants/
    COMMAND rsync -avz --delete ${CMAKE_SOURCE_DIR}/systemd/dev-orin.service root@dev-orin:/etc/systemd/system

    DEPENDS ${ALL_BUILD_TARGETS}
    
    COMMENT "Uploading folders to dev-orin..."
    VERBATIM
)
