infile  = open("data/coefs/L3_coefs_nodup.csv")
outfile = open("data/coefs/L3_coefs_proc_nodup.csv","w")

coefs_dict = {}

for line in infile:
	split_line = line.strip().split("\t")

	mods      = "_".join(split_line[0].split("_")[:-2])
	mods_feat = "_".join(split_line[1].split("_")[:-1])
	score     = float(split_line[2])

	if mods in coefs_dict.keys():
		if mods_feat in coefs_dict[mods].keys():
			coefs_dict[mods][mods_feat] += score
		else:
			coefs_dict[mods][mods_feat] = score
	else:
		coefs_dict[mods] = {}
		if mods_feat in coefs_dict[mods].keys():
			coefs_dict[mods][mods_feat] += score
		else:
			coefs_dict[mods][mods_feat] = score

for k,it in coefs_dict.items():
	for k2,it2 in it.items():
		print(k,it,k2,it2)
		outfile.write("%s\t%s\t%s\n" % (k,k2,it2))

outfile.close()

infile  = open("data/coefs/L3_coefs_dup.csv")
outfile = open("data/coefs/L3_coefs_proc_dup.csv","w")

coefs_dict = {}

for line in infile:
	split_line = line.strip().split("\t")

	mods      = "_".join(split_line[0].split("_")[:-2])
	mods_feat = "_".join(split_line[1].split("_")[:-1])
	score     = float(split_line[2])

	if mods in coefs_dict.keys():
		if mods_feat in coefs_dict[mods].keys():
			coefs_dict[mods][mods_feat] += score
		else:
			coefs_dict[mods][mods_feat] = score
	else:
		coefs_dict[mods] = {}
		if mods_feat in coefs_dict[mods].keys():
			coefs_dict[mods][mods_feat] += score
		else:
			coefs_dict[mods][mods_feat] = score

for k,it in coefs_dict.items():
	for k2,it2 in it.items():
		print(k,it,k2,it2)
		outfile.write("%s\t%s\t%s\n" % (k,k2,it2))

outfile.close()