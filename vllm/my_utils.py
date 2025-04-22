from torch.profiler import record_function
import functools

def profile_function(name=None):
    def decorator(func):
        func_name = getattr(func, '__name__', getattr(getattr(func, '__func__', None), '__name__', 'unknown'))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args and hasattr(args[0], '__class__'):
                full_name = name or f"{args[0].__class__.__name__}-{func_name}"
            else:
                full_name = name or func_name
            with record_function(full_name):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


def decorate_all_methods(decorator_factory):
    def decorate(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                decorated = decorator_factory()(attr_value)
            elif isinstance(attr_value, staticmethod):
                original_func = attr_value.__func__
                decorated = staticmethod(decorator_factory()(original_func))
            elif isinstance(attr_value, classmethod):
                original_func = attr_value.__func__
                decorated = classmethod(decorator_factory()(original_func))
            else:
                continue
            setattr(cls, attr_name, decorated)
        return cls
    return decorate
