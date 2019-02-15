from configs import *

with open(PRE_LIST, 'r') as inp:
	fns = [line.split(' ')[0] for line in inp.readlines()]

with open(PRESENT_LIST, 'w') as out:
	for fn in fns:
		out.write(fn + '\n')
