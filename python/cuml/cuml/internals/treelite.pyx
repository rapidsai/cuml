def safe_treelite_call(res: int, err_msg: str) -> None:
    cdef str err_msg_from_treelite
    if res < 0:
        err_msg_from_treelite = TreeliteGetLastError().decode("UTF-8")
        raise RuntimeError("{err_msg}\n{err_msg_from_treelite}")
