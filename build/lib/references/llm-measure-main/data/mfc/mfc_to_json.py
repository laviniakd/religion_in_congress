import json
import re
import pandas as pd


def read_json(input_filename):
    with open(input_filename) as input_file:
        data = json.load(input_file)
    return data

def get_text(data, article):
    return data[article]['text'].split("\n\n")[2]+"\n\n"+" ".join(data[article]['text'].split("\n\n")[3:])

def get_headline(data, article):
    headline_frame = data[article]['headline_frame']
    headline_start = -1
    headline_end = -1
    framing_data = data[article]['annotations']['framing']
    for annotators in framing_data:
        for codes in framing_data[annotators]:
            if codes['code'] == headline_frame:
                headline_start = codes['start']
                headline_end = codes['end']
    if headline_start >-1 and headline_end > -1:
        headline = data[article]['text'][headline_start:headline_end].strip()
    else:
        headline = data[article]['text'].split("\n\n")[2]
    return headline
        
def main():
    path = "/data/madesai/mfc_v4.0/immigration/immigration_labeled.json"
    data = read_json(path)
    codes = read_json("/data/madesai/mfc_v4.0/codes.json")

    out_data = {'Headline':{},'Headline_frame':{},"Text":{},'ID':{},'Label':{},'Year':{},'Month':{}}
    for article in data.keys():
    #text = data[article]['text']
    id = article
    headline = get_headline(data, article)
    if data[article]['irrelevant'] == 1:
        frame_number = 0
        headline_frame_number = 0
        print(id)
    else:
        frame_number = data[article]['primary_frame']
        headline_frame_number = data[article]['headline_frame']
        
    if frame_number !=0 and not frame_number:
        frame_number = 0
        headline_frame_number = 0

    headline_frame = re.sub(" headline", "", codes[str(frame_number)])
    frame =  codes[str(frame_number)]
    year = data[article]['year']
    month = data[article]['month']
    
    out_data['ID'][id] = id
    out_data['Headline'][id] = headline
    out_data['Headline_frame'][id] = headline_frame
    out_data['Text'][id] = get_text(data,article)
    out_data['Label'][id] = frame
    out_data['Year'][id] = year
    out_data['Month'][id] = month
    out_df = pd.DataFrame(out_data)
    out_df.to_json("/home/madesai/llm-measure/data/mfc/all_immigration.json")
if __name__ == "__main__":
    main()
    
          