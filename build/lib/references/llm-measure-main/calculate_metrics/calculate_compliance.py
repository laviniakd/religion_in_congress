import compliance_functions as cf
import pandas as pd
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="calculate-compliance")
    parser.add_argument("--task",
                        type = str)
    parser.add_argument("--filename",
                        type=str,)
    parser.add_argument("--label_space",
                        type=str,
                        default=None)
    parser.add_argument("--lower_bound",
                        type=int,
                        default=None,
                        help = "for scaled labels, the lower acceptable bound")
    parser.add_argument("--upper_bound",
                        type= int,
                        default=None,
                        help = "for scaled labels, the upper acceptable bound")
    parser.add_argument("--nruns",
                        type=int,
                        default=1)
    parser.add_argument("--pad_tokens",
                        type=str,
                        default= None)
    parser.add_argument("--grouped_on",
                        type=str,
                        default=None)
    parser.add_argument("--char",
                        type=int,
                        default=None,
                        help="if you just want to take the first char characters of the response, use this")
    parser.add_argument("--findall",
                        type=bool)
    args = parser.parse_args()
    
    return args





def main():
    args = parse_arguments()

    filename = args.filename
    task = args.task
    basepath = "/home/madesai/llm-measure/logs/"+task+"/"+filename
    data = pd.read_csv(basepath+"/responses.csv",index_col=0)
    grouped_on = args.grouped_on
   

    nruns = args.nruns
    pad_tokens = args.pad_tokens


    run_compliance = []
    for i in range(nruns):
    
        responses = list(data["response_"+str(i)])
        compliant_responses = []
        clean_responses = []

        for r in responses:
            if args.char:
                r = r[:args.char]
            if type(r) == str:
                r = r.strip()
       
            if pad_tokens: # remove pad tokens if applicable 
                r = str(r)
                r = re.sub(pad_tokens,"", r)
                r = re.sub("\|", "",r)
            if args.label_space: # if labels
                if "txt" in args.label_space:
                    label_space = ""
                    with open(args.label_space,"r") as f:
                        label_space = f.read().split("\n")
                else:
                    label_space = args.label_space.split(",")

                if args.char:
                    c = 1
                else:
                    c = None
                if args.findall:
                    compliance_test = cf.multiple_choice(label_space,r,leng=c, findall=True) # calculate compliance 
                else:
                    compliance_test = cf.multiple_choice(label_space,r,leng=c) # calculate compliance 
                
            if args.upper_bound or args.lower_bound: # if numerical within some bound 
                lower_bound = float(args.lower_bound) if args.lower_bound else None
                upper_bound = float(args.upper_bound) if args.upper_bound else None
                compliance_test = cf.scale(lower_bound,upper_bound,r) # calculate compliance 

            if compliance_test:
                compliant_responses.append(1)
                clean_responses.append(compliance_test) 
            else:
                compliant_responses.append(0)
                clean_responses.append(r)

        data["cleaned_responses_"+str(i)] = clean_responses
        data["compliant_responses_"+str(i)] = compliant_responses
        data = data.loc[:,~data.columns.str.match("Unnamed")]


        compliance = compliant_responses.count(1)/len(compliant_responses)

        run_compliance.append(compliance)

    if grouped_on:
        compliance_df = pd.DataFrame({"run":[i for i in range(nruns)], "compliance": run_compliance,"group":["all"]*nruns})
        for i in range(nruns):
            for item in data[grouped_on].unique():
                n_compliant = len(data.loc[(data[grouped_on]==item) & (data['compliant_responses_'+str(i)]==1)])
                n_total = len(data.loc[data[grouped_on]==item])
                compliance = n_compliant/n_total
                compliance_df.loc[len(compliance_df.index)] = [i, compliance, item] # add group data to df
    else:
        compliance_df = pd.DataFrame({"run":[i for i in range(nruns)], "compliance": run_compliance})
        

    print(basepath)
    compliance_df.to_csv(basepath+"/compliance_f1.csv")
    data.to_csv(basepath+"/responses.csv")

    


        
        

if __name__ == "__main__":
    main()
    