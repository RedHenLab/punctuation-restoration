import re

inf = open("russian_2019.txt", "r", encoding="utf-8")
outf= open("whole_russian_2019.txt", "w", encoding="utf-8")

lines = []

for l in inf.readlines():
    if re.match("^\s*<", l):
        continue
    else:
        l = l.split("\t")
        lines.append(l[0])

inf.close()

i = 1

for line in range(len(lines)-1):
    if re.match("^\s*[,;]" , lines[i]):
        lines[line] = lines[line] + "\tCOMMA"
    elif re.match("^\s*[\.:]" , lines[i]):
        lines[line] = lines[line] + "\tPERIOD"
    elif re.match("^\s*!*\?+\!*\?*" , lines[i]):
        lines[line] = lines[line] + "\tQUESTION" #!?, ???, !?! ...
    elif re.match("^\s*!+" , lines[i]):
        lines[line] = lines[line] + "\tEXCLAMATION"
    elif re.match("^\s*[\-+]" , lines[i]):
        lines[line] = lines[line] + "\tDASH"
    else:
        lines[line] = lines[line] + "\tO"
    i += 1

for le in lines:
    if re.match("^([\-\.,;!\?])+", le):
        continue
    outf.write(le + "\n")
