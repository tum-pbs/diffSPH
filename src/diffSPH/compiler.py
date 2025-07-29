
from pathlib import Path
import subprocess
import sys
import glob
import os
from typing import Optional
import platform
import torch
from torch.utils.cpp_extension import load
import warnings
import importlib

if torch.cuda.is_available():
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

directory = Path(__file__).resolve().parent

def find_cuda_home():
    """
    Finds the CUDA home directory by checking various possible locations.
    Based on the original script from PyTorch
    
    Returns:
        str: The path to the CUDA home directory, or None if it is not found.
    """
    IS_WINDOWS = sys.platform == 'win32'

    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
#             print('.', which)
            nvcc = subprocess.check_output(
                [which, 'nvcc'], env = dict(PATH='%s:%s/bin' % (os.environ['PATH'], sys.exec_prefix))).decode().rstrip('\r\n')
#             print(nvcc)
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    # if cuda_home and not torch.cuda.is_available():
        # print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    if cuda_home is not None:
        os.environ['CUDA_HOME'] = cuda_home
    # print('Cuda compiler:', cuda_home)
    return cuda_home

find_cuda_home()

def getComputeCapability(device):
    """
    Get the compute capability of the specified CUDA device.

    Args:
        device (int): The index of the CUDA device.

    Returns:
        int: The compute capability of the device.

    """
    return int(''.join([str(s) for s in torch.cuda.get_device_capability(device)]))



def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))

def _get_build_directory(name: str, verbose: bool) -> str:
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        cu_str = ('cpu' if torch.version.cuda is None else
                  f'cu{torch.version.cuda.replace(".", "")}')  # type: ignore[attr-defined]
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}'
        build_folder = f'{python_version}_{cu_str}'

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder)

    if verbose:
        print(f'Using {root_extensions_directory} as PyTorch extensions root...', file=sys.stderr)

    build_directory = os.path.join(root_extensions_directory, name)
    # if not os.path.exists(build_directory):
    #     if verbose:
    #         print(f'Creating extension directory {build_directory}...', file=sys.stderr)
    #     # This is like mkdir -p, i.e. will also create parent directories.
    #     os.makedirs(build_directory, exist_ok=True)

    return build_directory

def build_cpp_standard_arg(cpp_standard):
    """
    Build the argument for the C++ standard based on the given cpp_standard.
    Arguments are in the form of 'c++17'.

    Args:
        cpp_standard (str): The desired C++ standard.

    Returns:
        str: The argument for the C++ standard based on the platform.
    """
    if platform.system() == "Windows":
        return "/std:" + cpp_standard
    else:
        return "-std=" + cpp_standard

def compileSourceFiles(sourceFiles, module_name, directory: Optional[str] = None, 
                verbose = True, additionalFlags = [""], 
                openMP : bool = False, tbb : bool = False,
                verboseCuda : bool = False, cpp_standard : str = "c++17", cuda_arch : Optional[int] = None):
    """
    Compiles the given source files into a module.

    Args:
        sourceFiles (List[str]): List of source file paths.
        module_name (str): Name of the module.
        directory (Optional[str], optional): Directory path where the source files are located. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        additionalFlags (List[str], optional): Additional compilation flags. Defaults to [""].
        openMP (bool, optional): Whether to enable OpenMP support. Defaults to False.
        verboseCuda (bool, optional): Whether to print verbose output for CUDA. Defaults to False.
        cpp_standard (str, optional): C++ standard version. Defaults to "c++17".
        cuda_arch (Optional[int], optional): CUDA architecture version. Defaults to None.

    Returns:
        torch.utils.cpp_extension.CppExtension: Compiled module.
    """
    # verbose = False
    cpp_standard_arg = build_cpp_standard_arg(cpp_standard)

    hostFlags = [cpp_standard_arg, "-fPIC", "-O3", "-fopenmp", "-march=native"] if openMP else [cpp_standard_arg, "-fPIC", "-O3", "-march=native"]
    cudaFlags = [cpp_standard_arg,'-O3']
    
    if torch.cuda.is_available():
        computeCapability = getComputeCapability(torch.cuda.current_device()) if cuda_arch is None else cuda_arch
        if verbose:
            print('computeCapability:', computeCapability)
        smFlag = '-gencode=arch=compute_%d,code=sm_%d' % (computeCapability, computeCapability)
        # cudaFlags.append(smFlag)
        # cudaFlags.append('-arch=all -Wno-deprecated-gpu-targets -t 2')
        cudaFlags.append('-arch=native -Wno-deprecated-gpu-targets -t 1 --extended-lambda')
        cudaFlags.append('-allow-unsupported-compiler')
        if verbose:
            print('smFlag:', smFlag)
        cudaFlags.append('--use_fast_math')

        cudaFlags.append('-DCUDA_VERSION')
        hostFlags.append('-DCUDA_VERSION')
    if platform.system() == "Darwin":
        ldFlags = ['-fopenmp'] if openMP else []
    else:
        ldFlags = []
    if verboseCuda:
        cudaFlags.append('--ptxas-options="-v "')

    if verbose:
        print('hostFlags:', hostFlags)
        print('cudaFlags:', cudaFlags)
    if tbb:
        cudaFlags.append('-DTBB_VERSION')
        hostFlags.append('-DTBB_VERSION')

    if openMP:
        # clang under macos does not support fopenmp so check for existence of clang via homebrew
        # will fail if no clang is found
        if platform.system() == "Darwin":
            if not os.path.exists('/opt/homebrew/opt/llvm/bin/clang'):
                warnings.warn('No clang compiler found in homebrew installation. OpenMP support will not be available.')
                openMP = False
            else:
                os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'] if 'LDFLAGS' in os.environ else '', '-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++')
                os.environ['PATH'] = '%s %s' % ('/opt/homebrew/opt/llvm/bin:$PATH"', os.environ['PATH'])
                os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'] if 'LDFLAGS' in os.environ else '', "-L/opt/homebrew/opt/libomp/lib")
                os.environ['CPPFLAGS'] = '%s %s' % (os.environ['CPPFLAGS'] if 'CPPFLAGS' in os.environ else '', "-I/opt/homebrew/opt/llvm/include")
                os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'], "-L/opt/homebrew/opt/libomp/lib")
                os.environ['CPPFLAGS'] = '%s %s' % (os.environ['CPPFLAGS'], "-I/opt/homebrew/opt/llvm/include:-I/opt/homebrew/opt/libomp/include")
                os.environ['CC'] = '/opt/homebrew/opt/llvm/bin/clang'
                os.environ['CXX'] = '/opt/homebrew/opt/llvm/bin/clang'
                nvcc = subprocess.check_output(
                    ['which', 'clang'], env = dict(PATH='%s:%s/bin' % (os.environ['PATH'], sys.exec_prefix))).decode().rstrip('\r\n')
    if openMP:    
        cudaFlags.append('-DOMP_VERSION')
        hostFlags.append('-DOMP_VERSION')
    if directory is None:
        directory = Path(__file__).resolve().parent
    if verbose:
        print('directory:', directory)


    for sourceFile in sourceFiles:
        if verbose:
            print('sourceFile:', sourceFile)
        if os.path.exists(sourceFile) or os.path.exists(os.path.join(directory, sourceFile)):
            if verbose:
                print('source file exists:', sourceFile)
            continue
        else:
            raise RuntimeError('source file does not exist:', sourceFile)
    sourceFiles = [os.path.abspath(os.path.join(directory, sourceFile)) if os.path.exists(os.path.join(directory, sourceFile)) else sourceFile for sourceFile in sourceFiles]
    cppFiles = [sourceFile for sourceFile in sourceFiles if sourceFile.endswith('.cpp')]
    cuFiles = ['"%s"' % (sourceFile) for sourceFile in sourceFiles if sourceFile.endswith('.cu')]

    if verbose:
        print('cppFiles:', cppFiles)
        print('cuFiles:', cuFiles)

    if 'MAX_JOBS' not in os.environ:
        os.environ['MAX_JOBS'] = '16'

    version = platform.python_version()

    

    if torch.cuda.is_available():
        variant = platform.system() + '_py' + ''.join(version.split(".")[:-1]) + '_torch' + torch.__version__.split("+")[0].replace(".", "") + '_cu' + torch.version.cuda.replace(".", "")
        filepath = os.path.join(directory, 'prebuilt/') + variant + '.so'
        if verbose:
            print('Looking for prebuilt module:', filepath)
        if os.path.exists(filepath):
            if verbose:
                print('Loading:', variant)
            # warnings.warn(f'Loading {variant}.')
            # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(module)
            return module
        # warnings.warn(f'No prebuilt binary exists for the current configuration {variant}.')
        # warnings.warn('No prebuilt module found.')
        if not os.path.exists(_get_build_directory(module_name, verbose)):
            warnings.warn(f'No prior extension directory exists, fully recompiling code. (This may take a while.) Building for {variant}')
        return load(name=module_name, 
            sources=sourceFiles, verbose=verbose, extra_cflags=hostFlags, extra_cuda_cflags=cudaFlags, extra_ldflags=ldFlags)
    else:
        variant = platform.system() + '_py' + ''.join(version.split(".")[:-1]) + '_torch' + torch.__version__.split("+")[0].replace(".", "") + '_cpu'
        filepath = os.path.join(directory, 'prebuilt/') + variant + '.so'
        if verbose:
            print('Looking for prebuilt module:', filepath)
        if os.path.exists(filepath):
            if verbose:
                print('Loading:', variant)
            # warnings.warn(f'Loading {variant}.')
            # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(module)
            return module
        # warnings.warn(f'No prebuilt binary exists for the current configuration {variant}.')
        # warnings.warn('No prebuilt module found.')
        # verbose = True
        if not os.path.exists(_get_build_directory(module_name, verbose)):
            warnings.warn(f'No prior extension directory exists, fully recompiling code. (This may take a while.) Building for {variant}')        
        return load(name=module_name, 
            sources=cppFiles, verbose=verbose, extra_cflags=hostFlags, extra_ldflags=ldFlags)
