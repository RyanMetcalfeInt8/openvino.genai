echo "------------ Script initiatialization ------------"
set BUILD_LOC=C:\onnxruntime-genai-dev-tools\build_tools\BuildArtifacts-20250624_145252
set GENAIINTERFACES_LOC=%BUILD_LOC%\genai-interfaces
set OV_LOC=%BUILD_LOC%\openvino_2025.3.0.dev20250708_x86_64
set OVGENAI_LOC=%BUILD_LOC%\openvino.genai

echo "------------ Building GenAI Interfaces ------------"
cd %GENAIINTERFACES_LOC%
cmake -B build -S .
cmake --build build
cmake --install build --prefix %BUILD_LOC%/genai_interfaces_install
set genai_interfaces_DIR=%BUILD_LOC%\genai_interfaces_install\lib\cmake

echo "------------ Building openvino.genai ------------"
call %OV_LOC%\setupvars.bat || exit /b 1
mkdir %BUILD_LOC%\openvino.genai-build
cd %BUILD_LOC%\openvino.genai-build
cmake %OVGENAI_LOC%
cmake --build . --config Release
cmake --install . --prefix installed
set OpenVINOGenAI_DIR=%cd%\installed\runtime\cmake

echo "------------ Building openvino.genai Sample ------------"
mkdir %BUILD_LOC%\ov_sample_build
cd %BUILD_LOC%\ov_sample_build
cmake -DOVGENAI=ON %OVGENAI_LOC%\samples\cpp\genai_sample
cmake --build . --config Release
