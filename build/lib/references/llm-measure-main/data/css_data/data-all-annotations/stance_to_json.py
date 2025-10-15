import json
import pandas as pd

def read_json(input_filename):
    with open(input_filename) as input_file:
        data = json.load(input_file)
    return data

def main():
    path = "/home/madesai/llm-measure/data/css_data/data-all-annotations/"
    files = ["trialdata-all-annotations.txt", "trainingdata-all-annotations.txt", "testdata-taskA-all-annotations.txt", "testdata-taskB-all-annotations.txt"]
    
    for f in files:
        df = pd.read_csv(path+f,sep="\t", encoding='unicode_escape')
        df = df.loc[df['Opinion towards']=="TARGET"]
        ids = df['ID']
        df = df.set_index('ID')
        df['ID'] = ids
        df.to_json(path+f[:-4]+".json")
    
    df1 = pd.read_csv(path+files[0],sep="\t", encoding='unicode_escape')
    df2 = pd.read_csv(path+files[1],sep="\t", encoding='unicode_escape')
    df3 = pd.read_csv(path+files[2],sep="\t", encoding='unicode_escape')
    df4 = pd.read_csv(path+files[3],sep="\t", encoding='unicode_escape')

    all_df = pd.concat([df1, df2,df3,df4], axis=0)

    all_df.loc[all_df.Target == "Climate Change is a Real Concern" , 'Target'] = "the idea that climate change is a real concern"
    all_df.loc[all_df.Target == "Legalization of Abortion" , 'Target'] = "the legalization of abortion"
    all_df.loc[all_df.Target == "Feminist Movement" , 'Target'] = "the Feminist Movement"

    label_list = []
    for index, row in all_df.iterrows():
        if row['Opinion towards'] == "TARGET":
            label_list.append(row['Stance'])
        else:
            label_list.append("NONE")
    
    all_df.insert(6, "Label",label_list)
    ids = all_df['ID'].to_list()
    all_df = all_df.set_index('ID')
    all_df['ID'] = ids
    all_df = all_df.drop_duplicates("ID")
    all_df.to_json(path+"all_data"+".json")

if __name__ == "__main__":
    main()