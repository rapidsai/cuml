# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import cuml

from cuml.testing.utils import ClassEnumerator


def test_class_enumerator():
    class SomeModule:
        class SomeClass(cuml.Base):
            pass

        class ExcludedClass(cuml.Base):
            pass

        class CustomConstructorClass(cuml.Base):
            def __init__(self, *, some_parameter):
                self.some_parameter = some_parameter

            def __eq__(self, other):
                return self.some_parameter == other.some_parameter

    module = ClassEnumerator(
        module=SomeModule,
        exclude_classes=[SomeModule.ExcludedClass],
        custom_constructors={
            "CustomConstructorClass": lambda: SomeModule.CustomConstructorClass(
                some_parameter=1
            )
        },
    )

    models = module.get_models()
    ref = {
        "SomeClass": SomeModule.SomeClass,
        "CustomConstructorClass": lambda: SomeModule.CustomConstructorClass(
            some_parameter=1
        ),
    }

    # Here we don't do `assert models == ref` because CustomConstructorClass is
    # a lambda.
    assert len(models) == len(ref) == 2
    assert models["SomeClass"] == ref["SomeClass"]
    assert (
        models["CustomConstructorClass"]() == ref["CustomConstructorClass"]()
    )


def test_class_enumerator_actual_module():
    module = ClassEnumerator(
        module=cuml.linear_model,
        exclude_classes=[
            cuml.LinearRegression,
            cuml.MBSGDClassifier,
            cuml.MBSGDRegressor,
        ],
        custom_constructors={
            "LogisticRegression": lambda: cuml.LogisticRegression(handle=1)
        },
    )
    models = module.get_models()
    ref = {
        "ElasticNet": cuml.ElasticNet,
        "Lasso": cuml.Lasso,
        "LogisticRegression": lambda: cuml.LogisticRegression(handle=1),
        "Ridge": cuml.Ridge,
    }

    assert (
        models["LogisticRegression"]().handle
        == ref["LogisticRegression"]().handle
    )
    models.pop("LogisticRegression")
    ref.pop("LogisticRegression")
    assert models == ref


def test_class_enumerator_empty_module():
    class EmptyModule:
        pass

    assert {} == ClassEnumerator(EmptyModule).get_models()


def test_class_enumerator_parameters():
    class SomeModule:
        class SomeClass(cuml.Base):
            def __eq__(self, other):
                return type(other) is type(self)

    models1 = ClassEnumerator(module=SomeModule).get_models()
    models2 = ClassEnumerator(
        module=SomeModule, exclude_classes=[SomeModule.SomeClass]
    ).get_models()
    models3 = ClassEnumerator(
        module=SomeModule,
        custom_constructors={"SomeClass": lambda: SomeModule.SomeClass()},
    ).get_models()

    assert models1 == {"SomeClass": SomeModule.SomeClass}
    assert models2 == {}
    assert (
        len(models3) == 1 and models3["SomeClass"]() == SomeModule.SomeClass()
    )
