from collections import Counter

infile = open("retmetfeatures.csv")

present = []
for line in infile:
	splitline = line.split(",")[3:]
	present.append(",".join(splitline))

pres = Counter(present)

present = [x for x,y in pres.items() if y > 1]

print present

infile = open("retmetfeatures.csv")
outfile = open("retmetfeaturesRem.csv","w")

for line in infile:
	splitline = line.split(",")[3:]
	splitline = ",".join(splitline)
	if splitline in present: 
		print line
		continue
	outfile.write(line)
outfile.close()