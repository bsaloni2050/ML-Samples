"""this file prints the output to a text file"""


import subprocess
with open("output.txt", "w+") as output:
    subprocess.call(["python", "MC.py"], stdout=output);
