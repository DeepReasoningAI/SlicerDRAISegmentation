cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DRAISegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/DeepReasoningAI/SlicerDRAISegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "DRAI Team (Deep Reasoning AI)")
set(EXTENSION_DESCRIPTION
  "AI-powered segmentation for Spine/Pelvis (CT) and Aorta–Iliac–Femoral Arteries (CTA). "
  "For Spine/Pelvis CT, we generate individual segmentation masks for each vertebra, as well as the pelvis and sacrum; for CTA, we segment all traceable arteries below the aortic arch. Moreover, we can separate metal."
  "Upload CT volumes to the DRAI cloud and receive high-quality, multi-label masks directly in 3D Slicer within minutes."
  "Built to handle complex cases—including bypass grafts, occuled arteries, extreme scoliosis, metal implants, and challenging pathology—with reliable accuracy."
  "All data is securely processed and deleted immediately after inference. Questions? Contact support@deepreasoningai.com."
)
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/DRAISegmentation/Resources/Icons/DRAISegmentation.jpg")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/Screenshots/UI.png" "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/Screenshots/SpinePelvis.png" "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/Screenshots/AIF.png")
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
