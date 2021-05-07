import argparse
import json
import os


import re


def json_minify(string, strip_space=True):
    '''
    Based on JSON.minify.js:
    https://github.com/getify/JSON.minify
    Contributers:
    - Pradyun S. Gedam (conditions and variable names changed)
    '''
    tokenizer = re.compile(r'"|(/\*)|(\*/)|(//)|\n|\r')
    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):

        if not (in_multi or in_single):
            tmp = string[index:match.start()]
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub('[ \t\n\r]+', '', tmp)
            new_str.append(tmp)

        index = match.end()
        val = match.group()

        if val == '"' and not (in_multi or in_single):
            escaped = re.search(r'(\\)*$', string[:match.start()])

            # start of string or unescaped quote character to end string
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
                in_string = not in_string
            index -= 1  # include " character in next catch
        elif not (in_string or in_multi or in_single):
            if val == '/*':
                in_multi = True
            elif val == '//':
                in_single = True
        elif val == '*/' and in_multi and not (in_string or in_single):
            in_multi = False
        elif val in '\r\n' and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):
            new_str.append(val)

    new_str.append(string[index:])
    content = ''.join(new_str)
    content = content.replace(",]", "]")
    content = content.replace(",}", "}")
    return content


def add_prefix(branch_name):
    if '/' not in branch_name:
        return "origin/"+branch_name
    else:
        return branch_name


def change_branch(branch_str: str):
    branches = [add_prefix(b) for b in branch_str.split(",")]
    with open("../asv.conf.json", "r") as f:
        ss = f.read()
        config_json = json.loads(json_minify(ss))
        config_json["branches"] = branches
    with open("../asv.conf.json", "w") as f:
        json.dump(config_json, f)


if __name__ == "__main__":
    if "BRANCH_STR" in os.environ:
        change_branch(os.environ["BRANCH_STR"])
