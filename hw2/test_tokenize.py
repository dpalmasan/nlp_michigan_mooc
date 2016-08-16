import re

s = "The/DET K/NOUN factor/NOUN ,/. a/DET term/NOUN used/VERB to/PRT denote/VERB the/DET rate/NOUN of/ADP heat/NOUN transmission/NOUN through/ADP a/DET material/NOUN (/X B.t.u./sq./X ft./X of/X material/hr./*0F./in./X of/X thickness/X )/X ranges/VERB from/ADP 0.24/NUM to/ADP 0.28/NUM for/ADP flexible/ADJ urethane/NOUN foams/NOUN and/CONJ from/ADP 0.12/NUM to/ADP 0.16/NUM for/ADP rigid/ADJ urethane/NOUN foams/NOUN ,/. depending/ADP upon/ADP the/DET formulation/NOUN ,/. density/NOUN ,/. cell/NOUN size/NOUN ,/. and/CONJ nature/NOUN of/ADP blowing/VERB agents/NOUN used/VERB ./."
words = []
tags = []
for m in re.findall(r'(\S+)/([\w.]+)', s):
    words.append(m[0])
    tags.append(m[1])

print words
print tags

