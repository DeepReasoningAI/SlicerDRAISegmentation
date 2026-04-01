cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DRAISegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/Deep-Reasoning/SlicerDRAISegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "DRAI Team (Deep Reasoning AI)")
set(EXTENSION_DESCRIPTION
  "AI-powered medical image segmentation for Spine/Pelvis and Aorta-Iliac-Femoral structures. "
  "CT volumes are uploaded to the DRAI cloud server for deep-learning inference and "
  "multi-label segmentation masks are returned directly into 3D Slicer."
)
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/Deep-Reasoning/SlicerDRAISegmentation/main/DRAISegmentation/Resources/Icons/DRAISegmentation.png")
set(EXTENSION_SCREENSHOTURLS
  "https://raw.githubusercontent.com/Deep-Reasoning/SlicerDRAISegmentation/main/Screenshots/screenshot1.png"
)
set(EXTENSION_DEPENDS "NA")
set(EXTENSION_BUILD_SUBDIRECTORY inner-build)

set(SUPERBUILD_TOPLEVEL_PROJECT inner)

#-----------------------------------------------------------------------------
# Extension dependencies
#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
mark_as_superbuild(Slicer_DIR)

#-----------------------------------------------------------------------------
# Extension modules
#-----------------------------------------------------------------------------
add_subdirectory(DRAISegmentation)

#-----------------------------------------------------------------------------
# Packaging
#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
