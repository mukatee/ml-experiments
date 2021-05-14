import json
import requests
import csv
import time

#https://wiki.mozilla.org/Bugzilla:REST_API
#https://bugzilla.readthedocs.io/en/latest/api/core/v1/bug.html#search-bugs
#https://bugzilla.mozilla.org/rest/bug?limit=10&product=Firefox&include_fields=summary,status,resolution,description,id,dupe_of,creator,op_sys,platform,priority,severity,is_open,actual_time,estimated_time,see_also

columns = ["id",
           "summary",
           "status",
           "resolution",
           "description",
           "dupe_of",
           #"creator",
           "op_sys",
           "platform",
           "priority",
           "component",
           "severity",
           "creation_time",
           "last_change_time",
           "keywords",
           "resolution",
           "is_open"]
columns_str = ",".join(columns)

offset = 35000
limit = 1000

while True:
    #url="https://bugzilla.mozilla.org/rest/bug?limit=10&product=Firefox&include_fields=summary,status,resolution,description,id,dupe_of,creator,op_sys,platform,priority,severity,is_open,actual_time,estimated_time"
    url=f"https://bugzilla.mozilla.org/rest/bug?limit={limit}&offset={offset}&product=Firefox&include_fields={columns_str}"
    #url = "https://bugzilla.mozilla.org/rest/bug/35?include_fields=summary,status,resolution,description"
    r = requests.get(url)
    r_json = json.loads(r.text)
    n_bugs = len(r_json["bugs"])

    for bug in r_json["bugs"]:
        row = []
        for col in columns:
            if col == "creator":
                creator = bug["creator_detail"]
                creator_id = creator["id"]
                creator_real_name = creator["real_name"]
                creator_name = creator["name"]
                creator_email = creator["email"]
                creator_nick = creator["nick"]
                row.append(creator_id)
                row.append(creator_real_name)
                row.append(creator_name)
                row.append(creator_email)
                row.append(creator_nick)
                continue
            else:
                val = bug[col]
            row.append(val)
        with open('output.csv','a+') as csv_file:
            wr = csv.writer(csv_file)
            wr.writerow(row)

    if n_bugs < limit:
        print(f"stopping download, n_bugs={n_bugs} != limit={limit}")
        break
    if n_bugs > limit:
        print(f"WARNING: too many values read? {n_bugs}")
    print(f"offset finished now {offset}")
    time.sleep(3)
    offset += limit
