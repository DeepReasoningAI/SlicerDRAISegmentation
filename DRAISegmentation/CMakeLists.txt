cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DRAISegmentation)

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

set(MODULE_NAME DRAISegmentation)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.jpg
)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)

cpack_add_component(${MODULE_NAME}_Runtime)
