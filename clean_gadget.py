import re


def clean_gadget(gadget):

    # Regular expression patterns for function and variable names
    rx_fun = re.compile(r"\b(function)\s+([_A-Za-z]\w*)\b")
    rx_var = re.compile(r"\b(var|let|const)\s+([_A-Za-z]\w*)\b")

    # Dictionary to map function and variable names to symbols
    fun_symbols = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    cleaned_gadget = []

    for line in gadget:
        # Remove comments
        line = re.sub(r"//.*", "", line)
        line = re.sub(r"/\*.*?\*/", "", line)

        # Replace function names with symbols
        for match in rx_fun.finditer(line):
            func_keyword, func_name = match.groups()
            if func_name not in fun_symbols:
                fun_symbols[func_name] = f"FUN{fun_count}"
                fun_count += 1
            line = re.sub(
                r"\b" + re.escape(func_name) + r"\b", fun_symbols[func_name], line
            )

        # Replace variable names with symbols
        for match in rx_var.finditer(line):
            var_keyword, var_name = match.groups()
            if var_name not in var_symbols:
                var_symbols[var_name] = f"VAR{var_count}"
                var_count += 1
            line = re.sub(
                r"\b" + re.escape(var_name) + r"\b", var_symbols[var_name], line
            )

        cleaned_gadget.append(line)

    return cleaned_gadget
