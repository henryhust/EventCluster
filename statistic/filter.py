import re

with open("../data/lgbd/彩虹后宫_out_co.txt", "r", encoding="utf8") as fr:
    for line in fr.readlines():
        result = re.findall(r"@(.*)@", line)
        if result:
            print(result)
