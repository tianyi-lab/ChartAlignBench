import json

def data_alignment_str_to_json2(predicted_alignment, num_cell_difference):

    predicted_alignment = predicted_alignment.replace('\n', '')
    predicted_alignment = predicted_alignment[(predicted_alignment.find('[')) : (predicted_alignment.rfind(']') + 1)]
    print(predicted_alignment)

    predicted_alignment_json = None
    try:
        # convert string to Python list
        predicted_alignment_list = json.loads(predicted_alignment)

        # convert list to indexed dictionary
        predicted_alignment_json = {i: item for i, item in enumerate(predicted_alignment_list)}
    except:
        dummy_json_element = {"row name": "sample row", "column name": "sample column", "value in chart 1": float('inf'), "value in chart 2": float('inf')}
        predicted_alignment_json = {i: dummy_json_element for i in range(num_cell_difference)}

    return predicted_alignment_json
