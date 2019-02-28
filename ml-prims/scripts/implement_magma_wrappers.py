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

            while line is "\n":
                line = file.readline()

            while ';' not in line and line:
                curr_block.append(line)
                line = file.readline()
            curr_block.append(line)
            yield curr_block

    def reformat(self, curr_block):

        def reformat_def(block, typename):
            line = "".join(block)
            line = line.replace("*", "").replace('\n', " ")
            line = line.replace(" )", ")")
            fname = line[line.find(" "):line.find("(")]

            if typename is "float" :
                fname = fname.replace('z', "s")
            if typename is "double":
                fname = fname.replace('z', "d")

            inside = line[line.find("("):line.find(")") + 1]
            words = inside.split()

            # Keep only words with comma
            words = [" " + word for word in words if "," in word or ')' in word]
            words[-1] = words[-1].replace(",", "")
            words = ["{\n","return", fname, "("] + words + [";\n}\n"]
            new_line = "".join(words)
            return new_line

        def replace_to_type(typename):
            new_block = []
            for line in curr_block:
                for mapping in self.type_maps[typename] :
                    line = line.replace(mapping[0], mapping[1])
                new_block.append(line)
            if typename in ["float", "double"]:
                new_block[-1] = new_block[-1].replace(';', "")
            return new_block

        new_block = []
        new_block.append("template <typename T>\n")
        new_block += replace_to_type("template")

        new_block += ["\n", "template <>", "\n", "inline "]
        new_block += replace_to_type("float")
        new_block += ["\n"]
        new_block += reformat_def(curr_block, 'float')

        new_block += ["\n", "template <>", "\n", "inline "]
        new_block += replace_to_type("double")
        new_block += ["\n"]
        new_block += reformat_def(curr_block, 'double')
        new_block += ["\n"]

        return new_block

    @staticmethod
    def write_to_file(new_header, filename):
        with open(filename, 'w') as f:
            for item in new_header:
                f.write("%s" % item)

    def run(self, filename_in, filename_out):

        with open(filename_in, "r") as file :
            fun_generator = self.file_iterator(file)
            new_header = []

            for curr_block in fun_generator :
                new_block = self.reformat(curr_block)
                new_header += new_block

        self.write_to_file(new_header, filename_out)


if __name__ == "__main__":

    type_maps = {"template" : [["magmaDoubleComplex", 'T'],
                               ["z", '']],
                 "float": [["magmaDoubleComplex", 'float'],
                              ["z", '']],
                 "double": [["magmaDoubleComplex", 'double'],
                           ["z", '']]
                 }

    print("Please remove the commented lines from the header file")
    parser = MagmaParser(type_maps)
    parser.run(filename_in="temp/header.h",
               filename_out="temp/wrapped.h")
