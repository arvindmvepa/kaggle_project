import json


def remove_duplicates(lst):
    dup = set()
    new_lst = []
    for it in lst:
        json_it = json.dumps(it, sort_keys=True)
        if json_it not in dup:
            new_lst += [it]
            dup.add(json_it)
    return new_lst

