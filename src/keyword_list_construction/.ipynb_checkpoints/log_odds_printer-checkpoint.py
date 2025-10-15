import pandas as pd
import json

## Make log odds output slightly nicer

LOG_ODDS_PATH = '/data/laviniad/sermons-ir/log_odds/final/'
#SC_END = 'sermon_coca_odds.json'
#CS_END = 'coca_sermon_odds.json'
SC_END = 'sermon_congress_odds.json'
CS_END = 'congress_sermon_odds.json'
NUM = 250

with open(LOG_ODDS_PATH + CS_END) as f:
    d = json.load(f)
    
    for k, v in list(sorted(d.items(), key=lambda x: x[1]))[:NUM]:
        print("{:<8} {:<15}".format(k, v))
