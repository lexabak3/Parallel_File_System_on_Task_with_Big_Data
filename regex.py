import re


def replace_nth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string = before + after
    return new_string


lines =\
    """
83229            run.sh 2020-06-06T15:27:19 2020-06-06T16:26:30          cn-023        1   01:58:22 
83230            run.sh 2020-06-06T15:37:51 2020-06-06T16:39:25          cn-022        1   02:03:08 
    """
# print(lines, '\n')
lines = re.findall(r'\n\d.+', lines)
print(lines)
lines = [replace_nth(re.sub(r'[ ]+', '\t', line), '\t', '\t' * 3, 2) for line in lines]
print(''.join(lines))
