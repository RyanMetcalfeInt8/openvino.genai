This sample can be conditionally compiled to use either ORT GenAI or OpenVINO GenAI.


# Prepare GenAI Interfaces:

First step, is to prepare the GenAI Interfaces directory.

```
:: TODO: Change this once it's available to public
git clone https://github.com/intel-sandbox/genai-interfaces.git
cd genai-interfaces
cmake -B build -S .
cmake --build build
cmake --install build --prefix <your_preferred_path>/genai_interfaces_install
set genai_interfaces_DIR=<your_preferred_path>\genai_interfaces_install\lib\cmake
```

# ORT GenAI build

Pre-requisite here is that you have already built onnxruntime-genai & the C examples. (TODO: Add a link to the instructions)

First, within this `genai_sample` directory, prepare a `ort_genai/` folder that contains a `include` & `lib` folder that contains required collateral to build ORT GenAI samples.

For example, if you've already built ORT GenAI's C sample (phi3, etc.), then you probably have `include` & `lib` sitting in your `onnxruntime-genai\examples\c` directory.

In a cmd.exe shell, from this folder do:
```
mkdir ort_sample_build
cd ort_sample_build
cmake ..
cmake --build . --config Release
```

If all goes well, you should have `ort_sample_build\build\Release\chat_sample.exe`.

# OpenVINO GenAI build
*(Assuming Windows cmd.exe shell)*


Clone and build this fork of openvino-genai:
```
:: Download OpenVINO 2025.3 nightly build, extract it, set up env:
powershell -Command "Invoke-WebRequest -Uri "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.3.0-19447-f52861cec1e/openvino_toolkit_windows_2025.3.0.dev20250708_x86_64.zip" -OutFile openvino.zip
powershell -Command "Expand-Archive -LiteralPath 'openvino.zip' -DestinationPath 'openvino' -Force"
openvino\openvino_toolkit_windows_2025.3.0.dev20250708_x86_64\setupvars.bat

:: clone, build, install this fork / branch of openvino.genai:
git clone --recursive https://github.com/RyanMetcalfeInt8/openvino.genai.git --branch onnx_genai
mkdir openvino.genai-build
cd openvino.genai-build
cmake ..\openvino.genai
cmake --build . --config Release
cmake --install . --prefix installed
set OpenVINOGenAI_DIR=%cd%\installed\runtime\cmake
```

Build this sample:
```
mkdir ov_sample_build
cd ov_sample_build
cmake -DOVGENAI=ON ..
cmake --build . --config Release
```

If all goes well, you should have `ov_sample_build\build\Release\chat_sample.exe`.