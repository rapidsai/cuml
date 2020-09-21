import contextlib


@contextlib.contextmanager
def context_manager_func(*args, **kwargs):

    try:
        ret_val = yield "Yield Value"
    finally:
        print("Cleanup")


@context_manager_func()
def decorated_func():
    print("Decorated func")


def test_with():

    print("In Func, before")
    with context_manager_func() as cm_output:

        print("In With")

        with context_manager_func():
           print("In With 2")

    print("Out of with")


def test_wrapped():

    print("In Func, before")

    decorated_func()

    print("Out of decorated func")
