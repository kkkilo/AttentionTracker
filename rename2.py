import os
path = 'C:/Users/Tassi/PycharmProjects/Training_set/Inattentive'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, "inattentive." + str(i)+'.jpg'))
    i = i+1