@ECHO OFF
SETLOCAL EnableDelayedExpansion

CALL mkvirtualenv --system-site-packages %BUILD_TAG%
DEL /S /Q build
DEL /S /Q _download
MD build

SET _MSPDBSRV_ENDPOINT_=%BUILD_TAG%
SET TMP=%WORKSPACE%\\tmp
SET TEMP=%WORKSPACE%\\tmp
SET TMPDIR=%WORKSPACE%\\tmp

PUSHD build
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DUSE_OPENMP=ON -DBUILD_TORCH=ON -Dgtest_force_shared_crt=ON -DDMLC_FORCE_SHARED_CRT=ON -DBUILD_CPP_TEST=1 -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 17 2019" || EXIT /B 1
msbuild dgl.sln /m /nr:false || EXIT /B 1
COPY Release\dgl.dll .
COPY Release\runUnitTests.exe .
COPY tensoradapter\pytorch\Release\tensoradapter_pytorch*.dll tensoradapter\pytorch
POPD

PUSHD python
DEL /S /Q build *.egg-info dist
pip install -e . || EXIT /B 1
POPD

ENDLOCAL
EXIT /B
