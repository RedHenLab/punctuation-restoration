import re

inf = open("brown-family-unified.vrt", "r", encoding="utf-8")
outf= open("whole_brown-family-unified_original_features.txt", "w", encoding="utf-8")

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
    if re.match("^\s*[,:\-]" , lines[i]):
        lines[line] = lines[line] + "\tCOMMA"
    elif re.match("^\s*[;!\.]" , lines[i]):
        lines[line] = lines[line] + "\tPERIOD"
    elif re.match("^\s*!*\?+\!*\?*" , lines[i]):
        lines[line] = lines[line] + "\tQUESTION" #!?, ???, !?! ...
    else:
        lines[line] = lines[line] + "\tO"
    i += 1

for le in lines:
    if re.match("^([\-\.,;!\?])+", le):
        continue
    outf.write(le + "\n")
