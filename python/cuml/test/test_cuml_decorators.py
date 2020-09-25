from collections import deque
import contextlib
import typing
T = typing.TypeVar("T")


class BaseTest(object):
    def __init__(self) -> None:
        super().__init__()

        self.callbacks = deque()

    def process(self):
        for cb in self.callbacks:
            cb()

class Option1(BaseTest):
    def __init__(self) -> None:
        super().__init__()

        self.callbacks.append(self.my_func)

    def my_func(self):
        print("Option1 in my_func: {}".format(repr(self)))

class Option2(BaseTest):
    def __init__(self) -> None:
        super().__init__()

        self.callbacks.append(lambda: print("Option2"))

class Option3(BaseTest):
    def __init__(self) -> None:
        super().__init__()

        self.callbacks.append(lambda: print("Option3"))


class Combined(Option3, Option2, Option1):
    pass

class TestClass(object):
    def __init__(self) -> None:

        print()

    @classmethod
    def __class_getitem__(cls, params):

        return TestClass()


@contextlib.contextmanager
def context_manager_func(*args, **kwargs):

    try:
        ret_val = yield "Yield Value"
    finally:
        print("Cleanup")


@context_manager_func()
def decorated_func():
    print("Decorated func")


# def test_with():

#     print("In Func, before")
#     with context_manager_func() as cm_output:

#         print("In With")

#         with context_manager_func():
#            print("In With 2")

#     print("Out of with")


# def test_wrapped():

#     print("In Func, before")

#     decorated_func()

#     print("Out of decorated func")

def test_class():
    my_class = Combined()

    my_class.process()