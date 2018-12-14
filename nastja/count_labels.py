import re
output = format('gold_labels.txt')

with open('test.json.txt', 'r') as json_data:
    with open(output, 'w') as out:
        sp = au = wk = cap = no = 0
        spouse = re.compile('"has_spouse"')
        author = re.compile('"author"')
        work = re.compile('"worked_at"')
        capital = re.compile('"capital"')
        data = json_data.readlines()
        for line in data:
            if spouse.search(line):
                sp +=1
                out.write("has_spouse"+'\n')
            elif author.search(line):
                au += 1
                out.write("author"+'\n')
            elif work.search(line):
                wk +=1
                out.write("worked_at"+'\n')
            elif capital.search(line):
                cap +=1
                out.write("capital"+'\n')
            else:
                no +=1

print("has_spouse = {}; author = {}; worked_at = {}; capital = {}; NO REL = {}". format(sp, au, wk, cap, no))
