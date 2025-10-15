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
                        help = "if comparing agreement between runs in the same file, number of runs"
                        default = 0)
    parser.add_argument("--grouped_on",
                        type = str,
                        default=None,
                        help = "if data is grouped, column name of groups")
    parser.add_argument("--group_value",
                        type = str,
                        default=None,
                        help= "for comparing one particular group, name of group")
    
    args = parser.parse_args()
    return args

def calc_kripp(all_resp, krip_metric):
    if krip_metric == "nominal":
        return krippendorff_alpha(all_resp, nominal_metric, missing_items = "*")
    elif krip_metric == "interval":
        return krippendorff_alpha(all_resp, interval_metric, missing_items = "*")

# appends data from df to all_resp in the form {index:value}
def get_formatted_responses(all_resp, data, kripp_dict, column_n=0):
    if kripp_dict: 
        all_resp.append({data['Index'][i]:kripp_dict[data["cleaned_responses_"+str(column_n)][i]] for i in range(len(data))})
    else:
        all_resp.append({data['Index'][i]:data["cleaned_responses_"+str(column_n)][i] for i in range(len(data))})


def write_results(basepath, task, dir1, dir2, kripp, self):
    path = os.path.join(basepath,task, dir1,"kripp.csv")
    if os.path.exists(path):
        append_write = "a"
    else:
        append_write = "w"

    with open(path,append_write) as f:
        if append_write == "w":
            f.write("comparison,ka\n")
        if self:
            f.write("self,{:.4f}\n".format(kripp))
        else: 
            f.write("{},{:.4f}\n".format(dir2,kripp))
    
    if self == False:
        path2 = os.path.join(basepath,task,dir2,"kripp.csv")
        if os.path.exists(path2):
            append_write = "a"
        else:
            append_write = "w"
        with open(path2,append_write) as f:
            if append_write == "w":
                f.write("comparison,ka\n")
            f.write("{},{:.4f}\n".format(dir1,kripp))

def compare_data(nruns, data1, kripp_dict, krip_metric,grouped_on, group, data2, f1_label_space, f2_label_space, f2_labels):
    all_resp = []
    kripp_list = []
    group_list = []
    grouped_resp = []

    if nruns: 
        for i in range(nruns):
            data1.loc[data1['compliant_responses_'+str(i)] == 0, "cleaned_responses_"+str(i)] = "*"
            get_formatted_responses(all_resp, data1, kripp_dict, column_n=i,)

            if group:
                group_data = data1.loc[(data1[grouped_on]==group)]
                get_formatted_responses(grouped_resp, group_data, kripp_dict, column_n=i)
        self = True

    else: # comparing between files 

        data1.loc[data1['compliant_responses_0'] == 0, "cleaned_responses_0"] = "*"
        data2.loc[data2['compliant_responses_0'] == 0, "cleaned_responses_0"] = "*"

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

        get_formatted_responses(all_resp, data1,kripp_dict) # get all data 
        get_formatted_responses(all_resp,data2, kripp_dict)
    
    kripp = calc_kripp(all_resp, krip_metric)
    group_list.append("all")
    kripp_list.append(kripp)

    if group:
        group_kripp = calc_kripp(grouped_resp, krip_metric)
        group_list.append(group)
        kripp_list.append(group_kripp)
        
    return group_list, kripp_list

def main():
    args = parse_arguments()
    task = args.task
    nruns = args.nruns
    f2_label_space = args.f2_label_space
    f2_labels = args.f2_labels
    grouped_on = args.grouped_on
    group = args.group_value
    if "|" in group:
        group =" ".join(group.split("|"))
        print(group)
    base_path = "/home/madesai/llm-measure/logs/"
    krip_metric = args.kripp_metric    


    # get data from file1
    f1_name = args.file1
    f1 =  base_path + task + "/"+f1_name+"/responses.csv"
    data1 = pd.read_csv(f1)

    # get data from file2
    if args.file2:
        f2_name = args.file2
        f2 = base_path +task+"/" +f2_name +"/responses.csv"
        data2 = pd.read_csv(f2)
    else:
        f2_name = None
    
    # if labels are non numeric, create a mapping between labels for f1 and numbers
    if args.f1_labels:
        f1_label_space = args.f1_labels.split(",")
        kripp_dict = {f1_label_space[i]:i for i in range(len(f1_label_space))}
    else:
        kripp_dict = None


    if grouped_on and not group:
        for group in data1[grouped_on].unique():
            compare_data(nruns, data1, kripp_dict, krip_metric,grouped_on, group, data2, f1_label_space, f2_label_space, f2_labels)
    else:
        compare_data(nruns, data1, kripp_dict, krip_metric,grouped_on, group, data2, f1_label_space, f2_label_space, f2_labels)
    






    write_results(base_path, task, f1_name, f2_name, kripp, self)


if __name__ == "__main__":
    main()