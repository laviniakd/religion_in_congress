import argparse
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def parse_arguments():
    parser = argparse.ArgumentParser(description="calculate-compliance")
    parser.add_argument("--task",
                        type = str)
    parser.add_argument("--filename",
                        type = str,)
    parser.add_argument("--label_space",
                        type = str,
                        default = None)
    parser.add_argument("--response_space",
                         type = str,
                         default = None)
    parser.add_argument("--nruns",
                        type = int,
                        default = 1)
    parser.add_argument("--grouped_on",
                        type = str,
                        default = None)
    parser.add_argument("--mse",
                        default=False,
                        help = "if added will calculate mse")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_arguments()
    filename = args.filename
    task = args.task
    basepath = "/home/madesai/llm-measure/logs/"+task+"/"+filename
    data = pd.read_csv(basepath+"/responses.csv")
    nruns = args.nruns
    grouped_on = args.grouped_on
    mse = args.mse
 

    if args.label_space: # map between labels and responses 
        if "txt" in args.label_space:
            label_space = ""
            with open(args.label_space,"r") as f:
                label_space = f.read().split("\n")
        else:
            label_space = args.label_space.split(",")
        print(label_space)

    if args.response_space:
        if "txt" in args.response_space:
                    response_space = ""
                    with open(args.response_space,"r") as f:
                        response_space = f.read().split("\n")
        else:
            response_space = args.response_space.split(",")
        print(response_space)
        print(len(label_space),len(response_space))
        label_dict = {response_space[i]:label_space[i] for i in range(len(response_space))}
    print(label_dict)


    metrics_df = pd.read_csv(basepath+"/compliance_f1.csv",index_col=0)
    metrics_df['f1'] = [-1]*len(metrics_df)
    if mse:
        metrics_df['mse'] = [-1]*len(metrics_df)
    
    

    for i in range(nruns):
        ids = list(data.loc[data['compliant_responses_'+str(i)] == 1,'Index'])

        labels = list(data.loc[data['compliant_responses_'+str(i)] == 1,'labels'])
        labels = [str(l) for l in labels]
        responses = list(data.loc[data['compliant_responses_'+str(i)] == 1,'cleaned_responses_'+str(i)])
        if args.label_space and args.response_space:
            converted_responses = [label_dict[str(r)] for r in responses]
        else: converted_responses = [str(r) for r in responses]
       # for i in range(len(converted_responses)):
       #     print(labels[i], converted_responses[i])

        f1 = f1_score(labels,converted_responses, labels =label_space, average='micro')


        if grouped_on:
            metrics_df.loc[(metrics_df['group'] == 'all') & (metrics_df['run'] == i), 'f1'] = f1
            for item in data[grouped_on].unique():
                labels = list(data.loc[(data[grouped_on]==item) & (data['compliant_responses_'+str(i)]==1), 'labels'])
                responses = list(data.loc[(data[grouped_on]==item) & (data['compliant_responses_'+str(i)] == 1),'cleaned_responses_'+str(i)])
                converted_responses = [label_dict[r] for r in responses]
                f1 = f1_score(labels,converted_responses, labels =label_space, average='micro')
                metrics_df.loc[(metrics_df['group'] == item) & (metrics_df['run'] == i), 'f1'] = f1
        else:
            metrics_df.loc[metrics_df['run'] == i, 'f1'] = f1
        if mse:
            
              # print(ids[i],float(converted_responses[i]))
            labels = [float(l) for l in labels]
            converted_responses = [float(r) for r in converted_responses]
            mse_value = mean_squared_error(labels,converted_responses)
            metrics_df.loc[metrics_df['run'] == i, 'mse'] = mse_value

        # labels = list(data.loc[data['compliant_responses_'+str(i)] == 1,'labels'])
        # responses = list(data.loc[data['compliant_responses_'+str(i)] == 1,'cleaned_responses_'+str(i)])
        
        # converted_responses = [label_dict[r] for r in responses]

        # # for j in range(len(labels)):
        # #     print(labels[j], converted_responses[j], responses[j])
        
        # f1 = f1_score(labels,converted_responses, labels =label_space, average='macro')
        # f1_run.append(f1)
    
    

    
   # metrics_df['f1'] = f1_run
    metrics_df.to_csv(basepath+"/compliance_f1.csv")


if __name__ == "__main__":
    main()
    