cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DRAISegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/DeepReasoningAI/SlicerDRAISegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "DRAI Team (Deep Reasoning AI)")
set(EXTENSION_DESCRIPTION
  "DRAI Segmentation and 3D Modeling provides AI-powered segmentation and 3D surface modeling for Spine/Pelvis CT and Aorta-Iliac-Femoral CTA. Generate multi-label segmentation masks for individual vertebrae, the pelvis, sacrum, arteries, bone, and metal implants using the supported models. Upload CT volumes to the DRAI cloud for inference, then automatically load the segmentation masks and generate 3D surface models in 3D Slicer for visualization and analysis. Built to handle complex cases, including bypass grafts, occluded arteries, extreme scoliosis, metal implants, and challenging pathology. All data is securely processed and deleted immediately after inference. Questions? Contact support@deepreasoningai.com."
)
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/DRAISegmentation/Resources/Icons/DRAISegmentation.jpg")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/DeepReasoningAI/SlicerDRAISegmentation/main/Screenshots/UI.png") 
set(EXTENSION_DEPENDS "NA")
#-----------------------------------------------------------------------------
# Extension dependencies
#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
#-----------------------------------------------------------------------------
add_subdirectory(DRAISegmentation)

#-----------------------------------------------------------------------------
# Packaging
#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
