#pragma once
namespace ML {
namespace experimental {
namespace fil {

/* Enum representing possible row-wise operations on output */
enum struct row_op : unsigned char {
  disable=0b00100000,
  softmax=0b01000000,
  max_index=0b10000000
};

/* Enum representing possible element-wise operations on output */
enum struct element_op : unsigned char {
  disable=0b00000000,
  signed_square=0b00000001,
  hinge=0b00000010,
  sigmoid=0b00000100,
  exponential=0b00001000,
  logarithm_one_plus_exp=0b00010000
};

}
}
}
