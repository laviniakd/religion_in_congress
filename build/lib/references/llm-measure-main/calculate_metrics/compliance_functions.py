import re
import math


def multiple_choice(labels, response, leng =None, findall=None):
    if leng and len(response) > leng:
        return None
    
    
    
    compliance_string = "|".join(labels)
    if findall:
        match = re.findall(compliance_string,response)
    else:
        match = re.match(compliance_string,response)
    if match:
        return match[0]
    else:
        return None

def scale(lower_limit, upper_limit, response):
    response = response.split(".")[0]
    try:
        if math.isnan(float(response)):
            return None
        if lower_limit and float(response) < lower_limit:
            return None
        elif upper_limit and float(response) > upper_limit:
             print(response)
             return None
        else:
            return response
    except:
        return None