"""Pre-computed (and recomputed) golden values for tests."""
import json
import os



class GoldenModule:
    """Hold golden values for a particular test module."""

    VALUES_DIRECTORY_NAME = "golden_values"

    def __init__(self, module_name, module_dir, golden_mode):
        """Encapsulate golden values for test module."""
        self.module_name = module_name
        self.module_dir = module_dir

        cache_path = os.path.join(self.module_dir,
                                  GoldenModule.VALUES_DIRECTORY_NAME,
                                  "golden_%s.json" % (self.module_name))
        print("Using golden values stored in ", cache_path)
        self.filename = cache_path

        self.recompute = (golden_mode == "recompute" or golden_mode == "check")
        # Raise an exception if golden values have changed
        self.check_compute = golden_mode == "check"
        # Write all values out to JSONs at the end
        self.save_values = golden_mode == "recompute"
        print("Golden Module for %s. Recompute? %s Check? %s Save? %s" %
              (self.module_name,
               self.recompute,
               self.check_compute,
               self.save_values))

        self.test_values = self._load_test_values()

    def _load_test_values(self):
        try:
            with open(self.filename) as f:
                return json.load(f)
        except Exception:
            if self.recompute:
                print("... recomputing golden values for %s ..." %
                      self.module_name)
                return {}
            else:
                raise

    def cleanup(self):
        if self.save_values:
            with open(self.filename, "w") as f:
                json.dump(self.test_values, f, indent=1, sort_keys=True)


class Golden:
    """Hold golden values for a single test."""

    def __init__(self, golden_module, test_name):
        self.module = golden_module
        self.test_name = test_name
        self.recompute = golden_module.recompute
        self.check_compute = golden_module.check_compute
        if self.test_name not in self.module.test_values:
            self.module.test_values[self.test_name] = {}
        self.values = self.module.test_values[self.test_name]

    def __getattr__(self, attr):
        if attr[-1] == "_":
            return self.values[attr[:-1]]
        else:
            try:
                return self.__getitem__(attr)
            except KeyError:
                raise AttributeError(attr)

    def __setattr__(self, attr, value):
        """Store a new attr=value pair in recompute mode, failing
        if there is a change from pre-stored golden values."""
        if attr[-1] == "_":
            attr_name = attr[:-1]
            if attr_name in self.values:
                old_value = self.values[attr_name]
                print("Old/new values for %s: %s / %s" % (
                    self.test_name, old_value, value))

                if self.check_compute:
                    self._assert_value_equality(old_value, value)
            self.values[attr_name] = value
        else:
            return super().__setattr__(attr, value)

    def _assert_value_equality(self, old_value, new_value):
        """Asserts that old_value and new_value are equal
        Non-scalar types will NOT have equality checked.
        TODO: use flexible definition of equality for arrays, approx, etc."""
        if np.isscalar(old_value) and np.isscalar(new_value):
            assert old_value == new_value
        else:
            print("Cannot compare %s to %s yet" % (
                str(type(old_value)), str(type(new_value))
                ))
