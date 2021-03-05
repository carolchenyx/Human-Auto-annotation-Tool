import os

path_name='/home/hkuit155/Downloads/yoga_rename/yoga_rename_24'
i=0
for item in os.listdir(path_name):#进入到文件夹内，对每个文件进行循环遍历
    os.rename(os.path.join(path_name,item),os.path.join(path_name,(str(i)+'.jpg')))#os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
    i+=1
print(i)