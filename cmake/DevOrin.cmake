include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

set(COS_DEV_ORIN_DEPLOY_ROOT "/root" CACHE STRING
    "Destination directory on dev-orin (use a separate directory for concurrent worktrees)")
if(DEFINED ENV{COS_DEV_ORIN_DEPLOY_ROOT} AND NOT "$ENV{COS_DEV_ORIN_DEPLOY_ROOT}" STREQUAL "")
    set(COS_DEV_ORIN_DEPLOY_ROOT "$ENV{COS_DEV_ORIN_DEPLOY_ROOT}")
endif()
if(NOT COS_DEV_ORIN_DEPLOY_ROOT MATCHES "^/[-A-Za-z0-9_/]+$")
    message(FATAL_ERROR "COS_DEV_ORIN_DEPLOY_ROOT must be an absolute path containing only letters, digits, slashes, underscores, and hyphens")
endif()

set(DEV_ORIN_SERVICE_COMMAND)
if(COS_DEV_ORIN_DEPLOY_ROOT STREQUAL "/root")
    set(DEV_ORIN_SERVICE_COMMAND COMMAND rsync -avz --delete
        ${CMAKE_SOURCE_DIR}/systemd/dev-orin.service root@dev-orin:/etc/systemd/system)
endif()

add_custom_target(dev-orin
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying to root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}..."
    
    COMMAND ssh root@dev-orin "mkdir -p '${COS_DEV_ORIN_DEPLOY_ROOT}'"

    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/tests/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/tests/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/main/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/main/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/examples/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/examples/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/tools/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/tools/
    COMMAND rsync -avz --delete ${CMAKE_BINARY_DIR}/lib/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/lib/
    COMMAND rsync -avz --delete ${CMAKE_SOURCE_DIR}/constants/ root@dev-orin:${COS_DEV_ORIN_DEPLOY_ROOT}/constants/
    ${DEV_ORIN_SERVICE_COMMAND}
    
    COMMENT "Uploading folders to dev-orin..."
    VERBATIM
)
