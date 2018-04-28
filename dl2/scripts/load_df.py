#see https://gist.github.com/xuf12/8d7ddd92ec143d448a1772c8a538a88d

def get_texts(path):
    rows_list = []   
    for idx, label in enumerate(CLASSES):
        print(f'working on {path}/{label}')
        for fname in (path/f'{label}').glob('*.*'):
            dict1 = {}
            text = fname.open('r').read()
            dict1.update({
                'text':text,
                'label':idx
            }) 
            rows_list.append(dict1)
        print(len(rows_list))
    df = pd.DataFrame(rows_list)
    return df
    
df = get_texts(PATH/'data_raw')