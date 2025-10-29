# units (million, $ etc removed from COLUMN NAME)
    # units: sometimes represented in heading, sometimes in value hence UNCERTAIN
def remove_units_in_brackets(val):

    print(f"val before unit removal: {val}")

    idx_start = val.find('(')
    if idx_start == int(-1):
        # bracket doesn't exist
        return val    

    if val[idx_start-1] == ' ':
        idx_start -= 1          # space, remove it

    idx_end = val.find(')') + 1
    modified_val = val[:idx_start]
    if idx_end < len(val):
        modified_val += val[idx_end:]
    # modified_val = re.sub('([^>]+)', '', val)

    print(f"val after unit removal: {modified_val}")
    return modified_val

def extract_json_from_data_alignment_res(predicted_alignment, num_cell_difference):

    predicted_alignment = predicted_alignment.replace('\n', '')
    predicted_alignment_json = {}

    for i in range(num_cell_difference):
        
        cell_change_json_element = None
        try:
            idx_start = predicted_alignment.find('{')
            idx_end = predicted_alignment.find('}') + 1
            json_part = predicted_alignment[idx_start:idx_end]

            # print(json_part)

            predicted_alignment = predicted_alignment.replace(json_part, "*", 1)
            cell_change_json_element = json.loads(json_part)

        except:
            cell_change_json_element = {}
            cell_change_json_element["row name"] = "sample row"
            cell_change_json_element["column name"] = "sample column"
            cell_change_json_element["value in chart 1"] = float('inf')
            cell_change_json_element["value in chart 2"] = float('inf')


        predicted_alignment_json[i] = cell_change_json_element
    
    return predicted_alignment_json

