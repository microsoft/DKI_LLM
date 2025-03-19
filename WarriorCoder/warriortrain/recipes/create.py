f = open("/home/aiscuser/fhw/data/warrior_train.json", "r+")
lines = f.readlines()[:500]
fs = open("/home/aiscuser/fhw/data/warrior_test.json", "w+")
for line in lines:
    fs.write(line)