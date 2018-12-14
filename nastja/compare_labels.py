with open('gold_labels.txt', 'r') as g:
    with open('test_labels.txt', 'r') as c:
        capFP = capFN = auFP = auFN = wkFP = wkFN = spFP = spFN = 0
        for gl in g:
            gl = gl.strip()
            cl = c.readline()
            cl = cl.strip()


            if gl != 'capital' and cl == 'capital':
                capFP +=1
            elif gl != 'author' and cl == 'author':
                auFP +=1
            elif gl != 'worked_at' and cl == 'worked_at':
                wkFP +=1
            elif gl != 'has_spouse' and cl == 'has_spouse':
                spFP +=1
                
            if gl == 'capital' and cl != 'capital':
                capFN +=1
            elif gl == 'author' and cl != 'author':
                auFN +=1
            elif gl == 'worked_at' and cl != 'worked_at':
                wkFN +=1
            elif gl == 'has_spouse' and cl != 'has_spouse':
                spFN +=1

print('False Positives: capital = {}; author = {}; worked_at = {}; has_spouse = {}'. format(capFP, auFP, wkFP, spFP))

print('False Negatives: capital = {}; author = {}; worked_at = {}; has_spouse = {}'. format(capFN, auFN, wkFN, spFN))
