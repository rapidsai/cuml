class MagmaParser :
    def __init__(self, type_maps, n_skip_first=0):
        self.type_maps = type_maps
        self.n_skip_first = n_skip_first

    def file_iterator(self, file):
        # Skip first lines
        for _ in range(self.n_skip_first):
            file.readline()

        for line in file :
            curr_block = []

            while ';' not in line and line:
                curr_block.append(line)
                line = file.readline()
            curr_block.append(line)
            yield curr_block

    def replace_types(self, curr_block):

        def replace_to_type(typename):
            new_block = []
            for line in curr_block:
                for mapping in self.type_maps[typename] :
                    line = line.replace(mapping[0], mapping[1])
                new_block.append(line)
            return new_block

        full_block = []
        for typename in ["template", "float", "double"] :
            if typename is "template" :
                new_block = ["template <typename T>"]
            else :
                new_block = ["template <>"]
            new_block += replace_to_type(typename)

            line_two = "inline " + new_block[2]
            new_block[2] = line_two
            new_block.append("\n")

            full_block =  full_block + new_block
        return full_block

    def merge_fun_defs(self):
        pass

    @staticmethod
    def write_to_file(new_header, filename):
        with open(filename, 'w') as f:
            for item in new_header:
                f.write("%s" % item)

    def run(self, filename_in, filename_out):

        with open(filename_in, "r") as file :
            fun_generator = self.file_iterator(file)
            new_header = []

            for fun_str in fun_generator :
                new_block = self.replace_types(fun_str)
                new_header = new_header + new_block

        self.write_to_file(new_header, filename_out)


if __name__ == "__main__":

    type_maps = {"template" : [["DoubleComplex", 'T'],
                               ["z", '']],
                 "float": [["DoubleComplex", 'Float'],
                              ["z", 's']],
                 "double": [["DoubleComplex", 'Double'],
                           ["z", 'd']]
                 }
    parser = MagmaParser(type_maps)
    parser.run(filename_in="temp/sample_header.h",
               filename_out="temp/new_header.h")
