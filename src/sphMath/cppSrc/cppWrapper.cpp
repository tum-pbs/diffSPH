#include "common.h"
#include <gradientOperations/gradientVector.h>


#define PY_INIT_NAME(EXT) PyInit_##EXT
#define GLUE_INIT(...) PY_INIT_NAME(__VA_ARGS__)
#define PY_INIT_NAME_STR(EXT) #EXT
#define GLUE_NAME(...) PY_INIT_NAME_STR(__VA_ARGS__)

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* GLUE_INIT(TORCH_EXTENSION_NAME)(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          GLUE_NAME(TORCH_EXTENSION_NAME),   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

#define GLUE_LIBRARY(...) TORCH_LIBRARY(__VA_ARGS__, m)

namespace TORCH_EXTENSION_NAME{
// Create the python bindings for the C++ functions
  GLUE_LIBRARY(TORCH_EXTENSION_NAME) {
    m.def("gradientVector", &gradientVector);

  } 
}

