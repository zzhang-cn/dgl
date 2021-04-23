#!/usr/bin/env groovy

dgl_linux_libs = "build/libdgl.so, build/runUnitTests, python/dgl/_ffi/_cy3/core.cpython-36m-x86_64-linux-gnu.so, build/tensoradapter/pytorch/*.so"
// Currently DGL on Windows is not working with Cython yet
dgl_win64_libs = "build\\dgl.dll, build\\runUnitTests.exe, build\\tensoradapter\\pytorch\\*.dll"

def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule update --recursive --init"
}

def init_git_win64() {
  checkout scm
  bat "git submodule update --recursive --init"
}

// pack libraries for later use
def pack_lib(name, libs) {
  echo "Packing ${libs} into ${name}"
  stash includes: libs, name: name
}

// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  echo "Unpacked ${libs} from ${name}"
}

def build_dgl_linux(dev) {
  init_git()
  sh "bash tests/scripts/build_dgl.sh ${dev}"
  sh "ls -lh /usr/lib/x86_64-linux-gnu/"
  pack_lib("dgl-${dev}-linux", dgl_linux_libs)
}

def build_dgl_win64(dev) {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  init_git_win64()
  bat "CALL tests\\scripts\\build_dgl.bat"
  pack_lib("dgl-${dev}-win64", dgl_win64_libs)
}

def cpp_unit_test_linux() {
  init_git()
  unpack_lib("dgl-cpu-linux", dgl_linux_libs)
  sh "bash tests/scripts/task_cpp_unit_test.sh"
}

def cpp_unit_test_win64() {
  init_git_win64()
  unpack_lib("dgl-cpu-win64", dgl_win64_libs)
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 15, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend} ${dev}"
  }
}

def unit_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 10, unit: 'MINUTES') {
    bat "CALL tests\\scripts\\task_unit_test.bat ${backend}"
  }
}

def example_test_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_example_test.sh ${dev}"
  }
}

def example_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 20, unit: 'MINUTES') {
    bat "CALL tests\\scripts\\task_example_test.bat ${dev}"
  }
}

def tutorial_test_linux(backend) {
  init_git()
  unpack_lib("dgl-cpu-linux", dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_${backend}_tutorial_test.sh"
  }
}


pipeline {
  agent any
  stages {
    stage("Lint Check") {
      agent { 
        docker {
          label "linux-c52x-node"
          image "public.ecr.aws/s1o7b3d9/dgllib/dgl-ci-lint:latest"  
          alwaysPull true
        }
      }
      steps {
        init_git()
        sh "bash tests/scripts/task_lint.sh"
      }
      post {
        always {
          cleanWs disableDeferredWipeout: true, deleteDirs: true
        }
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent { 
            docker {
              label "linux-c52x-node"
              image "dgllib/dgl-ci-cpu:conda" 
              alwaysPull true
            }
          }
          steps {
            build_dgl_linux("cpu")
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("GPU Build") {
          agent {
            docker {
              label "linux-c52x-node"
              image "dgllib/dgl-ci-gpu:conda"
              args "-u root"
              alwaysPull true
            }
          }
          steps {
            // sh "nvidia-smi"
            build_dgl_linux("gpu")
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("CPU Build (Win64)") {
          // Windows build machines are manually added to Jenkins master with
          // "windows" label as permanent agents.
          agent { label "windows" }
          steps {
            build_dgl_win64("cpu")
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        // Currently we don't have Windows GPU build machines
      }
    }
    stage("Test") {
      parallel {
        stage("C++ CPU") {
          agent { 
            docker { 
              label "linux-c52x-node"
              image "dgllib/dgl-ci-cpu:conda"
              alwaysPull true
            }
          }
          steps {
            cpp_unit_test_linux()
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("C++ CPU (Win64)") {
          agent { label "windows" }
          steps {
            cpp_unit_test_win64()
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Tensorflow CPU") {
          agent { 
            docker {
              label "linux-c52x-node"
              image "dgllib/dgl-ci-cpu:conda" 
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("tensorflow", "cpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Tensorflow GPU") {
          agent { 
            docker { 
              label "linux-gpu-node"
              image "dgllib/dgl-ci-gpu:conda" 
              args "--runtime nvidia"
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("tensorflow", "gpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Torch CPU") {
          agent { 
            docker {
              label "linux-c52x-node"
              image "dgllib/dgl-ci-cpu:conda" 
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("pytorch", "cpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_linux("pytorch", "cpu")
              }
            }
            stage("Tutorial test") {
              steps {
                tutorial_test_linux("pytorch")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Torch CPU (Win64)") {
          agent { label "windows" }
          stages {
            stage("Unit test") {
              steps {
                unit_test_win64("pytorch", "cpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_win64("pytorch", "cpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Torch GPU") {
          agent {
            docker {
              label "linux-gpu-node"
              image "dgllib/dgl-ci-gpu:conda"
              args "--runtime nvidia"
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                sh "nvidia-smi"
                unit_test_linux("pytorch", "gpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_linux("pytorch", "gpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("MXNet CPU") {
          agent { 
            docker {
              label "linux-c52x-node"
              image "dgllib/dgl-ci-cpu:conda" 
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("mxnet", "cpu")
              }
            }
            //stage("Tutorial test") {
            //  steps {
            //    tutorial_test_linux("mxnet")
            //  }
            //}
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("MXNet GPU") {
          agent {
            docker {
              label "linux-gpu-node" 
              image "dgllib/dgl-ci-gpu:conda"
              args "--runtime nvidia"
              alwaysPull true
            }
          }
          stages {
            stage("Unit test") {
              steps {
                sh "nvidia-smi"
                unit_test_linux("mxnet", "gpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
      }
    }
  }
  post {
    always {
      node('windows') {
        bat "rmvirtualenv ${BUILD_TAG}"
      }
    }
  }
}
