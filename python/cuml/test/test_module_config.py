import cuml

from cuml.test.utils import ModuleConfig


def test_module_config():
    class SomeModule:
        class SomeClass(cuml.Base):
            pass

        class ExcludedClass(cuml.Base):
            pass

        class CustomConstructorClass(cuml.Base):
            def __init__(self, some_parameter):
                self.some_parameter = some_parameter

            def __eq__(self, other):
                return self.some_parameter == other.some_parameter

    module = ModuleConfig(module=SomeModule,
                          exclude_classes=[SomeModule.ExcludedClass],
                          custom_constructors={
                              "CustomConstructorClass":
                                  lambda: SomeModule.CustomConstructorClass(
                                      some_parameter=1)
                          })

    models = module.get_models()
    ref = {
        "SomeClass": SomeModule.SomeClass,
        "CustomConstructorClass": lambda: SomeModule.CustomConstructorClass(
            some_parameter=1)
    }

    # Here we don't do `assert models == ref` because CustomConstructorClass is
    # a lambda.
    assert len(models) == len(ref) == 2
    assert models['SomeClass'] == ref['SomeClass']
    assert models['CustomConstructorClass']() == ref[
        'CustomConstructorClass']()


def test_module_config_empty_module():
    class EmptyModule:
        pass

    assert {} == ModuleConfig(EmptyModule).get_models()


def test_module_config_parameters():
    class SomeModule:
        class SomeClass(cuml.Base):
            def __eq__(self, other):
                return type(other) == type(self)

    models1 = ModuleConfig(module=SomeModule).get_models()
    models2 = ModuleConfig(module=SomeModule,
                           exclude_classes=[SomeModule.SomeClass]).get_models()
    models3 = ModuleConfig(
        module=SomeModule,
        custom_constructors={'SomeClass': lambda: SomeModule.SomeClass()}
    ).get_models()

    assert models1 == {'SomeClass': SomeModule.SomeClass}
    assert models2 == {}
    assert len(models3) == 1 and models3[
        'SomeClass']() == SomeModule.SomeClass()
