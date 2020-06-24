def len_gt_n(word_strs, n):
    return word_strs.str.len() > n


def len_eq_n(word_strs, n):
    return word_strs.str.len() == n
