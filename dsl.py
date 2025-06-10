import toml
import sys


def transformToArgument(key, value, includeType = True, addUnderScore = False, includeOptional = False, functionOnly = False, typeFormat = 'pyBind', pyOnly = False):
    if not functionOnly and 'pythonArg' in value and value['pythonArg'] == False:
        return ""
    if not pyOnly and 'cppArg' in value and value['cppArg'] == True:
        return ""    
    if includeType:
        if typeFormat == 'pyBind':
            if 'tensor' in value['type']:
                type_str = f"torch::Tensor"
            else:
                type_str = f"{value['type']}"
            if 'optional' in value and value['optional']:
                type_str = f"std::optional<{type_str}>"
        elif typeFormat == 'compute':
            if 'tensor' in value['type']:
                ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                type_str = f"{'c' if 'const' not in value or value['const'] else ''}ptr_t<{ty}, {1 if 'dim' not in value else value['dim']}>"
            else:
                if value['type'] == 'double':
                    type_str = 'scalar_t'
                else:
                    type_str = f"{value['type']}"
            
    else:
        type_str = ""

    if addUnderScore:
        name_str = f"{key}_"
    else:
        name_str = f"{key}"

    if not includeOptional and 'optional' in value and value['optional']:
        return ""
    return f"{type_str} {name_str}"

def generateFunctionArguments(parsedToml, **kwargs):
    out = []

    for key, value in parsedToml.items():
        out.append(transformToArgument(key, value, **kwargs))

    out = [x for x in out if x != ""]



    return out
def generateTensorAccessors(parsedToml, optional = False):
    out = []
    for key, value in parsedToml.items():
        if 'cppArg' in value and value['cppArg'] == True:
            continue
        if optional: # only output optional values
            if 'optional' in value and value['optional']:
                if 'tensor' in value['type']:
                    ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                    dim = 1 if 'dim' not in value else value['dim']
                    out += [f"\tauto {key} = getAccessor<{ty}, {dim}>({key}_.value(), \"{key}\", useCuda, verbose_, true);\n"]
                else:
                    if value['type'] == 'double':
                        out += [f"\tauto {key} = (scalar_t) {key}_;\n"]
                    else:
                        out += [f"\tauto {key} = {key}_.value();\n"]
        else:
            if 'optional' in value and value['optional']:
                continue

            if 'tensor' in value['type']:
                ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                dim = 1 if 'dim' not in value else value['dim']

                out += [f"\tauto {key} = getAccessor<{ty}, {dim}>({key}_, \"{key}\", useCuda, verbose_);\n"]
            else:
                if value['type'] == 'double':
                    out += [f"\tauto {key} = (scalar_t) {key}_;\n"]
                else:
                    out += [f"\tauto {key} = {key}_;\n"]
    return out
import os
def process(fileName):
    filePrefix = fileName.split('.')[0].split('/')[-1]

    with open(fileName, 'r') as f:
        lines = f.readlines()

        # print(lines)
    try:
        tomlBegin = lines.index('/** BEGIN TOML\n')
    except:
        print(f"Could not find Description in {fileName}, skipping file")
        return
    print(f"Generating bindings for {fileName} with prefix {filePrefix}")
    tomlBegin = lines.index('/** BEGIN TOML\n')
    tomlEnd = lines.index('*/ // END TOML\n')
    tomlDefinitions = ''.join(lines[tomlBegin + 1: tomlEnd])

    if lines[tomlBegin - 1].startswith('/// functionName'):
        functionName = lines[tomlBegin - 1].split('=')[-1].strip()
        print(f"Function name: {functionName}")
        function = functionName.strip()
    else:
        function = filePrefix
        # tomlDefinitions = tomlDefinitions.replace('functionName', functionName)

    parsedToml = toml.loads(tomlDefinitions)


    endOfDefines = lines.index('/// End the definitions for auto generating the function arguments\n')

    prefixLines = lines[:tomlEnd+1]
    suffixLines = lines[endOfDefines:]

    if '// AUTO GENERATE ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// AUTO GENERATE ACCESSORS\n')
        accessorEnd = suffixLines.index('// END AUTO GENERATE ACCESSORS\n')
        accessorLines = generateTensorAccessors(parsedToml, optional = False)

        suffixLines = suffixLines[:accessorBegin+1] + accessorLines + suffixLines[accessorEnd:]
        finalLines = suffixLines[accessorEnd:]

    if '// AUTO GENERATE OPTIONAL ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// AUTO GENERATE OPTIONAL ACCESSORS\n')
        accessorEnd = suffixLines.index('// END AUTO GENERATE OPTIONAL ACCESSORS\n')
        accessorLines = generateTensorAccessors(parsedToml, optional = True)

        suffixLines = suffixLines[:accessorBegin+1] + accessorLines + suffixLines[accessorEnd:]
        finalLines = suffixLines[accessorEnd:]
        
    if '// GENERATE AUTO ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// GENERATE AUTO ACCESSORS\n')
        accessorEnd = suffixLines.index('// END GENERATE AUTO ACCESSORS\n')

        prefix = ['template<typename scalar_t = float>\n',
                  f'auto {function}_getFunctionArguments(bool useCuda, {function}_functionArguments_t){{\n']
        accessorLines = generateTensorAccessors(parsedToml, optional = False)
        accessorLinesOptional = generateTensorAccessors(parsedToml, optional = True)
        suffix = [f'\treturn std::make_tuple({function}_arguments_t);\n',
                  '}\n']
        suffixLines = suffixLines[:accessorBegin+1] + prefix + accessorLines + accessorLinesOptional+ suffix + suffixLines[accessorEnd:]
        finalLines = suffixLines[accessorEnd:]

    numOptionals = len([x for x in parsedToml.values() if 'optional' in x and x['optional']])
    print(f"Number of optional arguments: {numOptionals}")

    pyArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = False, typeFormat = 'pyBind', pyOnly = True))
    # fnArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = False, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
    # computeArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = False, functionOnly = True, typeFormat = 'compute', addUnderScore = False, pyOnly = False))
    arguments = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = False, pyOnly = False))
    arguments_ = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))

    # if numOptionals > 0:
    fnArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
    computeArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = True, typeFormat = 'compute', addUnderScore = False, pyOnly = False))
        # argumentsOptional = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = False, pyOnly = False))
        # argumentsOptional_ = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
    # else:
    #     fnArgumentsOptional = fnArguments
    #     computeArgumentsOptional = computeArguments
    #     argumentsOptional = arguments


    generatedLines = []
    generatedLines += ['\n', '// DEF PYTHON BINDINGS\n']
    generatedLines += [f'#define {function}_pyArguments_t {pyArguments}\n']
    generatedLines += ['// DEF FUNCTION ARGUMENTS\n']
    generatedLines += [f'#define {function}_functionArguments_t {fnArguments}\n']
    # if numOptionals > 0:
        # generatedLines += [f'#define {function}_functionArgumentsOptional_t {fnArgumentsOptional}\n']

    generatedLines += ['// DEF COMPUTE ARGUMENTS\n']
    generatedLines += [f'#define {function}_computeArguments_t {computeArguments}\n']
    # if numOptionals > 0:
    #     generatedLines += [f'#define {function}_computeArgumentsOptional_t {computeArgumentsOptional}\n']

    generatedLines += ['// DEF ARGUMENTS\n']
    generatedLines += [f'#define {function}_arguments_t {arguments}\n']
    # if numOptionals > 0:
        # generatedLines += [f'#define {function}_argumentsOptional_t {argumentsOptional}\n']
    generatedLines += [f'#define {function}_arguments_t_ {arguments_}\n']
    # if numOptionals > 0:
        # generatedLines += [f'#define {function}_argumentsOptional_t_ {argumentsOptional_}\n']

    generatedLines += ['\n', '// END PYTHON BINDINGS\n']


    outputTensors = []
    for key, value in parsedToml.items():
        if 'output' in value and value['output'] == True:
            print(f"Output tensor: {key}")
            outputTensors.append(key)

    if len(outputTensors) == 0:
        returnType = 'void'
    elif len(outputTensors) == 1:
        returnType = 'torch::Tensor'
    else:
        returnType = 'std::tuple<'
        for i in range(len(outputTensors)):
            returnType += f"torch::Tensor"
            if i != len(outputTensors) - 1:
                returnType += ", "
        returnType += ">"

    functionLines = []
    functionLines.append('namespace TORCH_EXTENSION_NAME {\n')
    functionLines.append(f'\t{returnType} {function}({function}_pyArguments_t);\n')
    functionLines.append('}\n')
    functionLines.append(f'void {function}_cuda({function}_functionArguments_t);\n')
    functionLines.append(f'template<std::size_t dim = 2, typename scalar_t = float>\n')
    functionLines.append(f'deviceInline auto {function}_impl(int32_t i, {function}_computeArguments_t){{\n')

    if '// AUTO GENERATE FUNCTIONS\n' in suffixLines:
        functionBegin = suffixLines.index('// AUTO GENERATE FUNCTIONS\n')
        functionEnd = suffixLines.index('// END AUTO GENERATE FUNCTIONS\n')
        # functionLines += suffixLines[functionBegin + 1: functionEnd]
        suffixLines = suffixLines[:functionBegin+1] + functionLines + suffixLines[functionEnd:]

    # suffixLines = suffixLines[:accessorBegin+1] + prefix + accessorLines + suffix + suffixLines[accessorEnd:]

    with open(fileName, 'w') as f:
        # f.writelines(prefixLines + ['#pragma region autoGenerated\n'] + generatedLines + ['#pragma endregion\n'] + suffixLines)
        f.writelines(prefixLines + generatedLines + suffixLines)


    # process cpp file

    filePrefix = fileName.split('.')[0].split('/')[-1]
    if not os.path.exists(fileName.split('.')[0] + "_cpu.cpp"):
        # create a new file
        with open(fileName.split('.')[0] + "_cpu.cpp", 'w') as f:
            f.writelines([
                '#include <algorithm>\n',
'#include <atomic>\n',
'#include <optional>\n',
'// AUTO GENERATED CODE BELOW\n',
'// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN\n',
'// AUTO GENERATED CODE BELOW - Part 2\n',
'// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN - Part 2\n',
'}\n'
            ])

    with open(fileName.split('.')[0] + "_cpu.cpp", 'r') as f:
        cpp_lines = f.readlines()

    autoGenBegin1 = cpp_lines.index('// AUTO GENERATED CODE BELOW\n')
    autoGenEnd1 = cpp_lines.index('// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN\n')
    autoGenBegin2 = cpp_lines.index('// AUTO GENERATED CODE BELOW - Part 2\n')
    autoGenEnd2 = cpp_lines.index('// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN - Part 2\n')

    prefixLines = cpp_lines[:autoGenBegin1]
    userLines = cpp_lines[autoGenEnd1 + 1: autoGenBegin2]
    finalLines = cpp_lines[autoGenEnd2 + 1:]

    preAmple = []
    preAmple += ['// AUTO GENERATED CODE BELOW\n']
    preAmple += [f'#include "{fileName.split('/')[-1]}"\n']
    preAmple += ['\n']
    
    preAmple += ['template <typename... Ts>\n',
                f'auto {function}_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, bool isCuda, Ts&&... args) {{\n',
                f'    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "{function}", [&]() {{\n',
                f'        auto functionArguments = invoke_bool({function}_getFunctionArguments<scalar_t>, isCuda, args...);\n',
                f'        parallelCall({function}_impl<dim_v, scalar_t>, 0, nQuery, functionArguments);\n',
                '    });\n',
                '}\n']
    
    referenceTensor = []
    for key, value in parsedToml.items():
        if 'reference' in value and value['reference'] == True:
            print(f"Reference tensor: {key}")
            referenceTensor.append(key)
    if len(referenceTensor) == 0:
        raise ValueError("No reference tensor found in toml file")
    

    preAmple += [f'torch::Tensor TORCH_EXTENSION_NAME::{function}({function}_pyArguments_t) {{\n',
        '// Get the dimensions of the input tensors\n',
        f'\tint32_t nQuery = {referenceTensor[0]}.size(0);\n',
        f'\tint32_t dim = {referenceTensor[0]}.size(1);\n',
        f'\tint32_t nSorted = {referenceTensor[0]}.size(0);\n',
    '\n',
        '// Create the default options for created tensors\n',
        f'\tauto defaultOptions = at::TensorOptions().device({referenceTensor[0]}.device());\n',
        '\tauto hostOptions = at::TensorOptions();\n',
    '\n',
    '// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN\n']

    postAmple = [
    '// AUTO GENERATED CODE BELOW - Part 2\n',
    f'\tauto wrappedArguments = std::make_tuple({function}_arguments_t);\n',
'\n',
f'    if ({referenceTensor[0]}.is_cuda()) {{\n',
'#ifndef WITH_CUDA\n',
'        throw std::runtime_error("CUDA support is not available in this build");\n',
'#else\n',
f'        std::apply({function}_cuda, wrappedArguments);\n',
'#endif\n',
'    } else {\n',
f'        {function}_cpu(nQuery, dim, {referenceTensor[0]}.scalar_type(), {referenceTensor[0]}.is_cuda(), wrappedArguments);\n',
'    }\n',
'\n',
'// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN - Part 2\n']
    # add the preAmple to the beginning of the file
    cpp_lines = prefixLines + preAmple + userLines + postAmple + finalLines

    with open(fileName.split('.')[0] + "_cpu.cpp", 'w') as f:
        f.writelines(cpp_lines)

    cuda_lines = [
f'#include "{function}.h"\n',
'\n',
f'void {function}_cuda({function}_functionArguments_t) {{\n',
f'    int32_t nQuery = {referenceTensor[0]}_.size(0);\n',
f'    auto scalar = {referenceTensor[0]}_.scalar_type();\n',
f'    auto dim = {referenceTensor[0]}_.size(1);\n',
'\n',
f'    auto wrappedArguments = std::make_tuple({referenceTensor[0]}_.is_cuda(), {function}_arguments_t_);\n',
'\n',
f'    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "{function}", [&]() {{\n',
f'        auto functionArguments = std::apply({function}_getFunctionArguments<scalar_t>, wrappedArguments);\n',
f'        launchKernel([] __device__(auto... args) {{ {function}_impl<dim_v, scalar_t>(args...); }}, nQuery, functionArguments);\n',
'    });\n',
'}\n'
    ]
    with open(fileName.split('.')[0] + "_cuda.cu", 'w') as f:
        f.writelines(cuda_lines)


if __name__ == "__main__":
    # only pass parameters to main()
    process(sys.argv[1:][0])