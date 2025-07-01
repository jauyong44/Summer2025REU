import importlib
import os
import inspect

def get_sever_method(args, cfg):
    method_name = cfg[args.method].global_method # Assuming 'global_method' is the key
    
    # Dynamically import *only the specific module requested*
    try:
        mod = importlib.import_module('Sever.' + method_name)
    except ImportError as e:
        raise ImportError(f"Could not import sever method '{method_name}'.")

    # Assuming the class name matches the module name
    class_obj = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if inspect.isclass(obj) and issubclass(obj, object) and 'SeverMethod' in str(inspect.getmro(obj)): # Check for SeverMethod in MRO
            if obj.__name__ != 'SeverMethod': # If SeverMethod itself might be imported, avoid returning it
                class_obj = obj
                break
    
    if class_obj is None:
        # Fallback for finding class name by iterating through module's members
        class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'SeverMethod' in str(
            inspect.getmro(getattr(mod, x))[1:])]
        if class_name:
            class_obj = getattr(mod, class_name[0])

    if class_obj is None:
        raise RuntimeError(f"Could not find a valid SeverMethod class in module 'Sever.{method_name}'")

    return class_obj(args, cfg) # Instantiate and return
