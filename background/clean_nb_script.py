with open('/mnt/963GB/Data/Python/Courses/fastai/my_fastai/dl2/scripts/translate.py') as infile, open('/mnt/963GB/Data/Python/Courses/fastai/my_fastai/dl2/scripts/translate_clean.py', 'w') as outfile:
    for line in infile :
        if not line.startswith('# In['):
            outfile.write(line)