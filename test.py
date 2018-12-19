file = open("result.txt", 'w')
for i in range(6):
    file.writelines(str(i) + "\n")