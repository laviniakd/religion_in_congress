from ka import interval_metric, krippendorff_alpha, nominal_metric
import pandas as pd
import argparse
import os.path


def parse_arguments():
    parser = argparse.ArgumentParser(description="calculate-compliance")
    parser.add_argument("--task",
                        type = str)
    parser.add_argument("--file1",
                        type = str,)
    parser.add_argument("--file2",
                        type=str)
    parser.add_argument("--f1_labels",
                        type = str,
                        help ="labels with no spaces, separated by commas e.g. if labels are A, B, or C, variable will be A,B,C",
                        default = None)
    parser.add_argument("--f2_labels",
                         type = str,
                         help ="labels with no spaces, separated by commas e.g. if labels are A, B, or C, variable will be A,B,C",
                         default = None)
    parser.add_argument("--kripp_metric",
                        type = str,
                        choices=["nominal","interval"])
    parser.add_argument("--nruns",
                        type = int,
                        help = "if comparing agreement between runs in the same file, number of runs",
                        default = 0)
    parser.add_argument("--grouped_on",
                        type = str,
                        default=None,
                        help = "if data is grouped, column name of groups")
    parser.add_argument("--group_value",
                        type = str,
                        default=None,
                        help= "for comparing one particular group, name of group")
    parser.add_argument("--labels",
                        type = str,
                        help = "if compare with labels, label column ID")
    
    args = parser.parse_args()
    return args

def calc_kripp(all_resp, krip_metric):
    if krip_metric == "nominal":
        return krippendorff_alpha(all_resp, metric=nominal_metric, missing_items = "*")
    elif krip_metric == "interval":
        return krippendorff_alpha(all_resp, interval_metric, missing_items = "*")
    
def calc_diff(all_resp):
    # [{id:resp}]
    answers1 = all_resp[0]
    answers2 = all_resp[1]
    disagree = 0
    count = 0
    for i in answers1:
        count +=1
        if answers1[i]!= answers2[i]:
            disagree+=1
    return disagree/count


def get_formatted_responses(all_resp, data, kripp_dict, column_n=0):
    cleaned_responses = list(data["cleaned_responses_"+str(column_n)])
    index = list(data["Index"])
    if kripp_dict: 
        print(kripp_dict)
        all_resp.append({index[i]:kripp_dict[cleaned_responses[i]] for i in range(len(cleaned_responses))})
        #all_resp.append({data['Index'][i]:kripp_dict[data["cleaned_responses_"+str(column_n)][i]] for i in range(len(data))})
    else:
        all_resp.append({index[i]:cleaned_responses[i] for i in range(len(cleaned_responses))})

        #all_resp.append({data['Index'][i]:data["cleaned_responses_"+str(column_n)][i] for i in range(len(data))})
    


def compare_data(args, data1, out_data, group, kripp_dict, data2=None):
    if args.f1_labels:
        if "txt" in args.f1_labels:
            f1_label_space = ""
            with open(args.f1_labels,"r") as f:
                f1_label_space = f.read().split("\n")
        else:
            f1_label_space = args.f1_labels.split(",")

    grouped_on = args.grouped_on

    f2_labels = args.f2_labels
    krip_metric = args.kripp_metric 

    all_resp = []
    grouped_resp = []
    if args.nruns >1: 
        for i in range(args.nruns):
            get_formatted_responses(all_resp, data1, kripp_dict, column_n=i,)
            if group:
                group_data = data1.loc[(data1[grouped_on]==group)]
         
                get_formatted_responses(grouped_resp, group_data, kripp_dict, column_n=i)

        out_data['file1'].append(args.file1)
        out_data['file2'].append(args.file1)

    else: # comparing between files
        

        if f2_labels: # map between two label spaces if they are different
            f2_label_space = f2_labels.split(",")
            if f2_label_space != f1_label_space: 
                label_dict = {f1_label_space[i]:f2_label_space[i] for i in range(len(f2_label_space))}
                label_dict["*"] = "*"
                data2.replace({"compliant_responses_0": label_dict})

        if group: # get group data
                group_data_1 = data1.loc[(data1[grouped_on]==group)]
                group_data_2 = data2.loc[(data2[grouped_on]==group)]
                get_formatted_responses(grouped_resp, group_data_1, kripp_dict)
                get_formatted_responses(grouped_resp, group_data_2, kripp_dict)

    if args.f1_labels: 
        out_data['file1'].append(args.file1)
        out_data['file2'].append(args.file2)

    if group:
        out_data['aspect'].append(group)
        group_kripp = calc_kripp(grouped_resp, krip_metric)
        group_diff = calc_diff(grouped_resp)
        out_data['ka'].append(group_kripp)
        out_data['percent_diff'].append(group_diff)
        print(group, group_kripp)
            

    if "all" not in out_data['aspect']:
        if args.nruns >1:
            out_data['file1'].append(args.file1)
            out_data['file2'].append(args.file1)
        else:
            out_data['file1'].append(args.file1)
            out_data['file2'].append(args.file2)
        get_formatted_responses(all_resp, data1,kripp_dict) # get all data 
        get_formatted_responses(all_resp,data2, kripp_dict)
        out_data['aspect'].append("all")
        kripp = calc_kripp(all_resp, krip_metric)
        diff = calc_diff(all_resp)
        print("all", kripp)
        out_data['ka'].append(kripp)
        out_data['percent_diff'].append(diff)
    return out_data
        
def write_results(base_path, task, dir, df):
    path = os.path.join(base_path,task, dir,"kripp.csv")
    if os.path.exists(path):
        append_write = "a"
    else:
        append_write = "w"

    df.to_csv(path, mode=append_write, header=not os.path.exists(path))

def main():
    args = parse_arguments()
    task = args.task
    nruns = args.nruns
    f2_labels = args.f2_labels
    grouped_on = args.grouped_on
    group = args.group_value
    if group and "|" in group:
        group =" ".join(group.split("|"))
        print(group)
    base_path = "/home/madesai/llm-measure/logs/"
    krip_metric = args.kripp_metric    

    # if labels are non numeric, create a mapping between labels for f1 and numbers
    if args.f1_labels:
        f1_label_space = args.f1_labels.split(",")
        kripp_dict = {f1_label_space[i]:i for i in range(len(f1_label_space))}
    else:
        kripp_dict = None


    # get data from file1
    f1_name = args.file1
    f1 =  base_path + task + "/"+f1_name+"/responses.csv"
    data1 = pd.read_csv(f1)
    data1.loc[data1['compliant_responses_0'] == 0, "cleaned_responses_0"] = "*"
    

    # get data from file2
    if args.file2:
        f2_name = args.file2
        f2 = base_path +task+"/" +f2_name +"/responses.csv"
        data2 = pd.read_csv(f2)
        data2.loc[data2['compliant_responses_0'] == 0, "cleaned_responses_0"] = "*"
        
    else:
        f2_name = None
        data2 = None

   
    out_data = {'file1':[],'file2':[],'aspect':[],'ka':[],'percent_diff':[]}

    if grouped_on and not group: #comparing across all groups
        for group in data1[grouped_on].unique():
            out_data = compare_data(args, data1, out_data, group, kripp_dict, data2)
    else:
        out_data = compare_data(args, data1, out_data, group, kripp_dict, data2)
    out_df = pd.DataFrame(out_data)
    write_results(base_path,task,f1_name, out_df)

    if f2_name:
        write_results(base_path,task,f2_name, out_df)

if __name__ == "__main__":
    main()