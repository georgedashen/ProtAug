import sys
import os

infile = sys.argv[1]
outfile = sys.argv[2]

os.system(f"sed 's/<eos>/L/g' {infile} > {outfile}")
os.system(f"sed -i 's/<cls>/L/g' {outfile}")
